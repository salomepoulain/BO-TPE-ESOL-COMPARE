
.PHONY: clean sim-results

clean:
	@echo "Cleaning up caches and build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@find . -type f -name "*~" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@rm -rf .mypy_cache .ruff_cache .pytest_cache
	@rm -rf src/*.egg-info build/ dist/
	@find . -type d \( -name ".matplotlib" -o -name "matplotlib-cache" \) -exec rm -rf {} +
	@echo "Done."

sim-results:
	@mkdir -p output
	@echo "Downloading simulation results..."
	@curl -L "https://amsuni-my.sharepoint.com/personal/salome_poulain_student_uva_nl/_layouts/15/download.aspx?share=IQBObImgeBweT7w5nSuRrFF0AUjBm5Ch43jPvHMVDIslOIY" -o output/simulation.zip
	@echo "Extracting results..."
	@unzip -q -o output/simulation.zip -d output/
	@rm output/simulation.zip
	@echo "Done. Results extracted to output/"
