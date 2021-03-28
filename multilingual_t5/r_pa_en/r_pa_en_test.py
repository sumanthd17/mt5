"""r_pa_en dataset."""

import tensorflow_datasets as tfds
from . import r_pa_en


class RPaEnTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for r_pa_en dataset."""
  # TODO(r_pa_en):
  DATASET_CLASS = r_pa_en.RPaEn
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
