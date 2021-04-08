"""baseline_or dataset."""

import tensorflow_datasets as tfds

# TODO(baseline_or): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(baseline_or): BibTeX citation
_CITATION = """
"""


class BaselineOr(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for baseline_or dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(baseline_or): Specifies the tfds.core.DatasetInfo object
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
    # TODO(baseline_or): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/ai4b-consolidated/strict-en-or.zip')

    # TODO(baseline_or): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=path/'en-or/train/train.or', target=path/'en-or/train/train.en'),
        'validation': self._generate_examples(source=path/'en-or/dev/dev.or', target=path/'en-or/dev/dev.en')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(baseline_or): Yields (key, example) tuples from the dataset
    src = tf.io.gfile.GFile(source, 'r').readlines()
    tgt = tf.io.gfile.GFile(target, 'r').readlines()
    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
          'source': row[0],
          'target': row[1],
      }
