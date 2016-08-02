generate_code:
	jupyter nbconvert --to script notebooks/1.0-wj-bayes-factors.ipynb
	mv notebooks/1.0-wj-bayes-factors.py src/models/bayes_factors.py
	jupyter nbconvert --to script notebooks/1.0-wj-trait-simulation.ipynb
	mv notebooks/1.0-wj-trait-simulation.py src/models/trait_simulation.py
	jupyter nbconvert --to script notebooks/1.0-wj-colocalisation.ipynb
	mv notebooks/1.0-wj-colocalisation.py src/models/colocalisation.py

environment:
	pip freeze > requirements.txt
	conda env export > environment.yaml

	
