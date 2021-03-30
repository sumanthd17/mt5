"""r_ic_all_ta dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(r_ic_all_ta): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(r_ic_all_ta): BibTeX citation
_CITATION = """
"""


class RIcAllTa(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for r_ic_all_ta dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(r_ic_all_ta): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'source': tfds.features.Text(),
            'target': tfds.features.Text(),
        }),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(r_ic_all_ta): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/ai4b-models/mT5/ta/ic_all_ta.zip')

    # TODO(r_ic_all_ta): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=path/'en-ta/train/train.en', target=path/'en-ta/train/train.ta'),
        'validation': self._generate_examples(source=path/'en-ta/dev/dev.en', target=path/'en-ta/dev/dev.ta')
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(r_ic_all_ta): Yields (key, example) tuples from the dataset
    src = tf.io.gfile.GFile(source, 'r').readlines()
    tgt = tf.io.gfile.GFile(target, 'r').readlines()
    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
          'source': row[0],
          'target': row[1],
      }
