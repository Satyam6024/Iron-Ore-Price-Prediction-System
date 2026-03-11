# Presentation Assets

Generated files:
- `Iron_Ore_Project_Presentation.pptx`
- `IRON_ORE_PROJECT_PRESENTATION.md`
- figures in `presentation/figures/`

Regenerate figures:

```powershell
cd C:\aditi
.\.venv\Scripts\Activate.ps1
python presentation/generate_presentation_assets.py
```

Regenerate the `.pptx` deck:

```powershell
cd C:\aditi
.\.venv\Scripts\Activate.ps1
uv pip install --python .\.venv\Scripts\python.exe python-pptx
python presentation/build_presentation.py
```
