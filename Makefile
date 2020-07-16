.PHONY: clean data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python3
DATE = $(shell date +%Y%m%d)
VERSION = v0.1.0

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Pre process train data
pre_process_train:
	$(PYTHON_INTERPRETER) pre-process.py \
						--input_path "data/train.csv" \
						--output_path "pruebas_j/train_clean.csv" \
						--set_ "train"

## Pre process test data
pre_process_test:
	$(PYTHON_INTERPRETER) pre-process.py \
						--input_path "data/test_santander.csv" \
						--output_path "pruebas_j/test_santander_clean.csv" \
						--set_ "test"

## Pre process data
pre_process: pre_process_train pre_process_test


## Translate train
translate_train:
	$(PYTHON_INTERPRETER) translate.py \
						--input_path "data/train.csv" \
						--output_path "data/train_translate.csv" \
						--set_ "train" \
						--pivot "en"

## Translate test
translate_test:
	$(PYTHON_INTERPRETER) translate.py \
						--input_path "data/test_santander.csv" \
						--output_path "data/test_translate.csv" \
						--set_ "test" \
						--pivot "en"

## Translate
translate: translate_train translate_test

## Train
train:
	$(PYTHON_INTERPRETER) train.py \
						--input_path "data/train_with_translations_clean.csv" \
						--output_dir "transformer_out_dir/"

## Predict
predict:
	$(PYTHON_INTERPRETER) predict.py \
						--input_path "data/test_with_translations_clean.csv" \
						--output_dir "transformer_out_dir/" \
						--tdd False


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
