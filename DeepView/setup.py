from setuptools import setup

setup(
	name='deepview', 
	packages=['deepview'],
	version='0.1.0',
	include_package_data=True,
	install_requires=[
		'matplotlib>=3.1.2',
		'scikit-learn',
		'scipy',
		'umap-learn>=0.3.10',
		'PyQt5>=5.13.2'
	],
	description='Implementation of the DeepView classifier-visualization framework',
	long_description='''Implementation of the DeepView framework, 
		as presented in this paper: https://arxiv.org/abs/1909.09154. 
		It can be used to visualize decision boundaries of classifiers that
		output class probabilities.''',
	url="https://github.com/LucaHermes/DeepView",
)