from setuptools import setup, find_packages

setup(
	name='document_similarity',
	version='0.0.1',
	python_requires='>=3.5',

	packages=find_packages(where='src'),
	package_dir={'': 'src'},

	install_requires=[
		'segtok-spans',
		'gensim==3.5.0',
		'numpy~=1.18',
		'pdftotext',
		'sphinx'
	],
	entry_points={
		'console_scripts': [
			'doc-sim=document_similarity.document_similarity:main',
		],
	},
)
#pip install git+ssh://git@github.com/emtelligent/segtok-spans.git
