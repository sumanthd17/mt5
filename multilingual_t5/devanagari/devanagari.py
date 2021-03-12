"""devanagari dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(devanagari): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(devanagari): BibTeX citation
_CITATION = """
"""

INDIC_LANGS = ['en', 'devanagari']

class DevanagariConfig(tfds.core.BuilderConfig):
  def __init__(self, name, languages, **kwargs):
    super().__init__(name=name, version=VERSION, **kwargs)

    self.languages = languages


class Devanagari(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for devanagari dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
    DevanagariConfig(
      "devanagari",
      languages=INDIC_LANGS,
      description=""
    )
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(devanagari): Specifies the tfds.core.DatasetInfo object
    features = {"text": tfds.features.Text()}
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(devanagari): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('pre-train/transliterated/data/transliterated.zip')

    # TODO(devanagari): Returns the Dict[split names, Iterator[Key, Example]]
    splits = []
    for lang in self.builder_config.languages:
      splits.extend(
        [
          tfds.core.SplitGenerator(
            name=lang, gen_kwargs=dict(path=/f"transliterated/{lang}.txt")
          ),
          tfds.core.SplitGenerator(
            name=f'{lang}-validation', gen_kwargs=dict(path=path/f"transliterated/{lang}-validation.txt"),
          )
        ]
      )
    return splits

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(devanagari): Yields (key, example) tuples from the dataset
    lines = tf.io.gfile.GFile(path, mode='r').readlines()

    for idx, row in enumerate(lines):
      yield idx, {
        'text': row
      }
