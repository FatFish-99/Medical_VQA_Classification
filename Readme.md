# Medical VQA Classification

Binary yes/no classification on medical images using Visual Question Answering.  
Dataset: [robailleo/medical-vision-llm-dataset](https://huggingface.co/datasets/robailleo/medical-vision-llm-dataset)  
Course: COSC 4368

---

## Requirements

- Python 3.10.x — download from [python.org](https://www.python.org/downloads/)
- VS Code with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Medical_VQA_Classification
```

### 2. Create the virtual environment

**Mac / Linux**
```bash
python3.10 -m venv .venv
```

**Windows**
```powershell
py -3.10 -m venv .venv
```

> If `py -3.10` is not found, use the full path instead:  
> `C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv`

### 3. Activate the virtual environment

**Mac / Linux**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
.venv\Scripts\activate.bat
```

> If PowerShell blocks the script, run this once to allow local scripts:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Register the Jupyter kernel

This step pins the kernel to the venv Python so Jupyter can never pick up a system Python by mistake.

**Mac / Linux**
```bash
.venv/bin/python -m ipykernel install --user --name medical_vqa --display-name "Python 3.10 (Medical VQA)"
```

**Windows**
```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name medical_vqa --display-name "Python 3.10 (Medical VQA)"
```

### 6. Open the notebook in VS Code

1. Open `medical_vqa_project.ipynb`
2. Click the kernel selector in the top-right corner of the notebook
3. Select **Python 3.10 (Medical VQA)**
4. Run the first cell to verify all imports succeed

---

## Project structure

```
Medical_VQA_Classification/
├── .venv/                     # virtual environment (not committed)
├── medical_vqa_project.ipynb  # main notebook
├── requirements.txt           # top-level dependencies
└── Readme.md
```

---

## Troubleshooting

**Kernel still uses wrong Python / numpy dtype error on import**  
The kernel was likely resolving to a system Python. Re-run step 5 to re-register it with the absolute venv path.

**`py -3.10` not recognized on Windows**  
Install Python 3.10 from [python.org](https://www.python.org/downloads/) and make sure "Add Python to PATH" is checked during installation.

**PowerShell execution policy error**  
Run the `Set-ExecutionPolicy` command from step 3, then try activating again.
