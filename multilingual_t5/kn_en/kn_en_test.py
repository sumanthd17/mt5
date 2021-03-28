"""kn_en dataset."""

import tensorflow_datasets as tfds
from . import kn_en


class KnEnTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for kn_en dataset."""
  # TODO(kn_en):
  DATASET_CLASS = kn_en.KnEn
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
