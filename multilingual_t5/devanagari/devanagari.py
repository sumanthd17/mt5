"""devanagari dataset."""

import tensorflow_datasets as tfds

# TODO(devanagari): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(devanagari): BibTeX citation
_CITATION = """
"""


class Devanagari(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for devanagari dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(devanagari): Specifies the tfds.core.DatasetInfo object
    features = {"text": tfds.features.Text()}
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage="https://indicnlp.ai4bharat.org/home/",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(devanagari): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.googleapis.com/pre-train/transliterated/data/devanagari.zip')

    # TODO(devanagari): Returns the Dict[split names, Iterator[Key, Example]]
    for lang in self.builder_config.languages:
      splits.extend(
        [
          tfds.core.SplitGenerator(
            name=lang, gen_kwargs=dict(path=path/f"devanagari/{lang}/", split='train', lang=lang)
          ),
          tfds.core.SplitGenerator(
            name=f'{lang}-validation', gen_kwargs=dict(path=path/f"devanagari/{lang}/", split='eval', lang=lang),
          )
        ]
      )
    return splits

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(devanagari): Yields (key, example) tuples from the dataset
    if split == 'eval':
      with open(str(path)+'/'+f'{lang}-validation.txt', 'r') as f:
        lines = f.readlines()
      for idx, row in enumerate(lines):
        yield idx, {
          'text': row
        }
    if split == 'train':
      count = 0
      for i, file_ in enumerate(os.listdir(path)):
        if 'val' in file_:
          continue
        with open(str(path)+'/'+file_, 'r') as f:
          lines = f.readlines()
        # print(i, file_)
        for idx, row in enumerate(lines):
          count += 1
          yield count, {
            'text': row
          }
