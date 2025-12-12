#region Imports
# Python standard library
from datetime import datetime, timedelta
from logging import DEBUG
from subprocess import run as run_subprocess, CompletedProcess
from pathlib import Path
from os import environ, getcwd
from random import randint


# Third party Packages
import nibabel
import SimpleITK as sitk
from typing import Optional
import numpy
import numpy as np
from rt_utils import RTStructBuilder
from dicomnode.dicom.dimse import Address
from dicomnode.server.grinders import ListGrinder
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import DicomOutput, PipelineOutput, FileOutput, MultiOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.lib.logging import get_logger

#region Environment Setup
ENVIRONMENT_ARCHIVE_PATH = "PIPELINE_ARCHIVE_PATH"
ENVIRONMENT_ARCHIVE_PATH_VALUE = environ.get(ENVIRONMENT_ARCHIVE_PATH,
                                             "/tmp/pet_gtx_pipeline_archive")

ARCHIVE_PATH = Path(ENVIRONMENT_ARCHIVE_PATH_VALUE)
if not ARCHIVE_PATH.exists():
  ARCHIVE_PATH.mkdir()

ENVIRONMENT_WORKING_PATH = "PIPELINE_WORKING_PATH"
ENVIRONMENT_WORKING_PATH_VALUE = environ.get(ENVIRONMENT_WORKING_PATH,
                                             "/tmp/pet_gtx_pipeline_working")

WORKING_PATH = Path(ENVIRONMENT_WORKING_PATH_VALUE)
if not WORKING_PATH.exists():
  WORKING_PATH.mkdir()


ENVIRONMENT_LOG_PATH = "PIPELINE_LOG_PATH"
ENVIRONMENT_LOG_PATH_VALUE = environ.get(ENVIRONMENT_LOG_PATH,
                                         "/var/log/pipeline")

LOG_PATH = Path(ENVIRONMENT_LOG_PATH_VALUE)


ENVIRONMENT_DCM2NIIX_PATH = "PIPELINE_DCM2NIIX"
DCM2NIIX = environ.get(ENVIRONMENT_DCM2NIIX_PATH,
                                             "dcm2niix")

which_output = run_subprocess(['which', DCM2NIIX], capture_output=True)
if(not len(which_output.stdout)):
  raise Exception("COULD NOT FIND DCM2NIIX")


ENVIRONMENT_RESAMPLE_PATH = "PIPELINE_RESAMPLE"
RESAMPLE = environ.get(ENVIRONMENT_RESAMPLE_PATH,
                                             "reg_resample")

which_output = run_subprocess(['which', RESAMPLE], capture_output=True)
if(not len(which_output.stdout)):
  raise Exception("COULD NOT FIND RESAMPLE program")

ENVIRONMENT_ACCEPTED_AE_TITLE = "PIPELINE_AE_TITLE"
RAW_AE_TITLES = environ.get(ENVIRONMENT_ACCEPTED_AE_TITLE,
                                             None)
if RAW_AE_TITLES is None:
  ae_titles = []
else:
  ae_titles = [
    ae_title.strip() for ae_title in RAW_AE_TITLES.split(",")
  ]

ENVIRONMENT_SEGMENTATION_PATH = "PIPELINE_SEGMENTATION_ARCHIVE"
RAW_SEGMENTATION_PATH = environ.get(ENVIRONMENT_SEGMENTATION_PATH,
                                                None)
if RAW_SEGMENTATION_PATH is not None:
  SEGMENTATION_PATH = Path(RAW_SEGMENTATION_PATH)
  if not SEGMENTATION_PATH.exists():
    raise Exception(f"Segmentation path {RAW_SEGMENTATION_PATH} does not exists, please create it!")
  if SEGMENTATION_PATH.is_file():
    raise Exception(f"Segmentation path {RAW_SEGMENTATION_PATH} should NOT be a file!")
else:
  SEGMENTATION_PATH = None

#region Setup
def crop_to_350_mm(nii_ct_path: Path, destination: Optional[Path] = None, crop_mm: float = 350.0) -> str:
  nii_ct_path = Path(nii_ct_path)

  if destination is None:
    destination = nii_ct_path.with_name("HNC04_000_CT.nii.gz")

  img = sitk.ReadImage(str(nii_ct_path))  # x,y,z in SimpleITK
  size_x, size_y, size_z = img.GetSize()
  spacing_x, spacing_y, spacing_z = img.GetSpacing()

  if spacing_z <= 0:
    raise ValueError(f"Invalid spacing_z={spacing_z} for {nii_ct_path}")

  slices_to_keep = int(numpy.ceil(crop_mm / spacing_z))
  slices_to_keep = max(1, min(slices_to_keep, size_z))

  # Keep the last slices (matches current downstream padding logic)
  start_z = size_z - slices_to_keep

  roi_index = [0, 0, start_z]
  roi_size  = [size_x, size_y, slices_to_keep]

  cropped = sitk.RegionOfInterest(img, size=roi_size, index=roi_index)
  sitk.WriteImage(cropped, str(destination))

  return str(destination)

def find_dcm2niix_output(cwd: Path, stem: str) -> Path:
  candidates = [
    cwd / f"{stem}.nii.gz",
    cwd / f"{stem}.nii",
  ]
  for p in candidates:
    if p.exists():
      return p
  raise FileNotFoundError(f"Could not find dcm2niix output for '{stem}' in {cwd}")


timestamp_format = "%Y%m%d%H%M%S.%f"

def dose_calculation(initial_dose,
                     halflife_seconds,
                     decay_time_delta: timedelta ,
                     ):
  return initial_dose * numpy.exp(numpy.log(2) / halflife_seconds * (-decay_time_delta.seconds))


def suv_rescale(image: numpy.ndarray, dose:float, patient_weight: float):
  return image / (dose / patient_weight)

output_address = Address(
  '10.49.144.12',
  104,
  'LILJEFORS',
)


#region Inputs
class PET_Input(AbstractInput):
  required_values = {
    0x00080060 : 'PT'
  }

  def validate(self) -> bool:
    return self.images > 0

  image_grinder = ListGrinder()

class CT_Input(AbstractInput):
  required_values = {
    0x00080060 : 'CT'
  }

  def validate(self) -> bool:
    return self.images > 0

  image_grinder = ListGrinder()

#region Pipeline
class PET_GTV_Pipeline(AbstractQueuedPipeline):
  input = {
    'PET' : PET_Input,
    'CT'  : CT_Input,
  }
  require_calling_aet = []

  study_expiration_days=1
  ip='0.0.0.0'
  port=11112
  log_output = Path(LOG_PATH)
  ae_title = "PETGTVAISEG"
  data_directory = ARCHIVE_PATH
  processing_directory = WORKING_PATH
  log_level = DEBUG

  def log_subprocess(self, output: CompletedProcess, process_name: str, log_anyways=False):
    if output.returncode != 0:
      self.logger.error(f"{process_name} return code: {output.returncode}")
      #self.logger.error(f"{process_name} stdout: {output.stdout.decode()}")
      #self.logger.error(f"{process_name} stderr: {output.stderr.decode()}")
      return
    if log_anyways:
      self.logger.info(f"{process_name} return code: {output.returncode}")
      #self.logger.info(f"{process_name} stdout: {output.stdout.decode()}")
      #self.logger.info(f"{process_name} stderr: {output.stderr.decode()}")

  def process(self, input_data: InputContainer) -> PipelineOutput:
    ct_path = input_data.paths['CT']
    pet_path = input_data.paths['PET']
    pivot_pet_dataset = input_data['PET'][0]

    # region SUV calculation
    patient_weight = pivot_pet_dataset.PatientWeight
    acquisition_date_str = pivot_pet_dataset.AcquisitionDate
    acquisition_time_str = pivot_pet_dataset.AcquisitionTime
    acquisition_datetime = datetime.strptime(f"{acquisition_date_str}{acquisition_time_str}",timestamp_format)
    tracer_info = pivot_pet_dataset.RadiopharmaceuticalInformationSequence[0]
    injection_datetime = datetime.strptime(tracer_info.RadiopharmaceuticalStartDateTime, timestamp_format)
    decay_delta_time = acquisition_datetime - injection_datetime
    injection_dose_MBq = tracer_info.RadionuclideTotalDose / 1_000_000
    halflife_seconds = tracer_info.RadionuclideHalfLife
    corrected_dose = dose_calculation(injection_dose_MBq, halflife_seconds, decay_delta_time)


    # Dicom to nifti conversion
    cwd = Path(getcwd())
    pet_destination_path = "HNC04_000_PET.nii.gz"
    ct_command = [DCM2NIIX, '-o', str(cwd), '-f', 'ct',str(ct_path)]
    self.log_subprocess(run_subprocess(ct_command, capture_output=True),
                        "dcm2niix ct")


    pet_command = [DCM2NIIX, '-o', str(cwd), '-f', 'pet', str(pet_path)]
    self.log_subprocess(run_subprocess(pet_command, capture_output=True),
                        "dcm2niix pet")

    #ct_nifti = nibabel.load('ct.nii')
    #data = ct_nifti.get_fdata().astype('float32')
    #header = ct_nifti.header.copy()
    #header.set_data_dtype('float32')
    #nibabel.save(nibabel.Nifti1Image(data, ct_nifti.affine, header), 'ct_f32.nii')

    ct_nii_path = find_dcm2niix_output(Path(getcwd()), "ct")
    ct_nifti_path = crop_to_350_mm(ct_nii_path, destination=Path(getcwd()) / "HNC04_000_CT.nii.gz")

    self.logger.info("Preprocessing step 1 complete, resampeling")
    #region Resampling
    resample_command = [
      'reg_resample',
      '-ref', ct_nifti_path,
      '-flo', 'pet.nii',
      '-res', pet_destination_path,
    ]

    self.log_subprocess(run_subprocess(resample_command, capture_output=False),
                        'Pet Resample')

    self.logger.info("Resampleing compelete")
    pet_image = nibabel.load(pet_destination_path)
    pet_data = pet_image.get_fdata()
    pet_data = suv_rescale(pet_data, corrected_dose, patient_weight)
    pet_image = nibabel.Nifti1Image(pet_data, pet_image.affine, pet_image.header)
    nibabel.save(pet_image, pet_destination_path)

    #
    segmentation_path = cwd / "segmentation.nii.gz"
    podman_command = ['podman',
                    'run',
                    '--security-opt=label=disable',
                    '--device=nvidia.com/gpu=all',
                    '-v',
                    f'{str(cwd)}:/usr/src/app/dataset',
                    'depict/hnc_pet_gtv:latest',
                    pet_destination_path,
                    ct_nifti_path,
                    "segmentation.nii.gz"
                  ]
    self.logger.info("Started podman process")
    self.log_subprocess(run_subprocess(podman_command, capture_output=False),                        'Podman',
                        log_anyways=False)
    self.logger.info("Finished podman process, started post processing")
    segmentation: nibabel.nifti1.Nifti1Image = nibabel.load(str(segmentation_path))

    #self.logger.error("Pet image affine")
    #self.logger.error(pet_image.affine)
    #self.logger.error("Segmentation image affine")
    #self.logger.error(segmentation.affine)

    if SEGMENTATION_PATH is not None:
      segmentation_path = SEGMENTATION_PATH / f"HNC07_{pivot_pet_dataset.PatientID}_{pivot_pet_dataset.StudyInstanceUID}.nii.gz"
      self.logger.info(f"Saved file at {segmentation_path}")
      segmentation.to_filename(segmentation_path)

    pipeline_mask = segmentation.get_fdata().astype(numpy.bool_)

    rotate_mask = numpy.rot90(pipeline_mask, 1, (0,1))

    # Resize mask such that fits with the CT
    empty_mask = numpy.zeros((pet_data.shape[0],
                              pet_data.shape[1],
                              len(input_data.datasets['CT']) - pet_data.shape[2]),
                              dtype=numpy.bool_)

    mask = numpy.concatenate((empty_mask, rotate_mask), axis=2)


    rt_struct = RTStructBuilder.create_new(
      str(ct_path)
    )

    rt_struct.add_roi(
      mask=mask,
      color=[255,255,255],
      name="PET GTV AI Segmentation",
      description="PET GTV AI Segmentation"
    )

    rt_dataset = rt_struct.ds
    # The output dataset to change
    rt_dataset.SeriesDescription = "PET GTV AI Segmentation"
    rt_dataset.SeriesNumber = randint(5000,100000)
    now = datetime.now()
    rt_dataset.SeriesTime = now.time()
    rt_dataset.SeriesDate = now.date()

    self.logger.info("Finished Post processing")

    return DicomOutput([
      (output_address, [rt_dataset]),
    ], self.ae_title)

  def post_init(self) -> None:
    cwd = getcwd()
    self.logger.info(f"Started to run the process at {cwd}")

#region __main__
if __name__ == '__main__':
  pipeline = PET_GTV_Pipeline()
  pipeline.open()
