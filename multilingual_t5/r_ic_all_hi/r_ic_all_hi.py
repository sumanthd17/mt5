"""r_ic_all_hi dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(r_ic_all_hi): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(r_ic_all_hi): BibTeX citation
_CITATION = """
"""


class RIcAllHi(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for r_ic_all_hi dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(r_ic_all_hi): Specifies the tfds.core.DatasetInfo object
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
    # TODO(r_ic_all_hi): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.cloud.google.com/ai4b-anuvaad-nmt/ai4b-models/mT5/hi/ic_all_hi.zip')

    # TODO(r_ic_all_hi): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=path/'en-hi/train/train.hi', target=path/'en-hi/train/train.en'),
        'validation': self._generate_examples(source=path/'en-hi/dev/dev.hi', target=path/'en-hi/dev/dev.en')
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(r_ic_all_hi): Yields (key, example) tuples from the dataset
    src = tf.io.gfile.GFile(source, 'r').readlines()
    tgt = tf.io.gfile.GFile(target, 'r').readlines()
    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
          'source': row[0],
          'target': row[1],
      }
