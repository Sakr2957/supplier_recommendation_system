# 🚀 Complete GitHub Deployment Guide

This guide will walk you through deploying your Smart Supplier Recommendation System to GitHub and Streamlit Cloud.

## 📋 Prerequisites

- GitHub account ([Sign up here](https://github.com/join))
- Git installed on your computer ([Download here](https://git-scm.com/downloads))
- GitHub CLI (optional but recommended) ([Install here](https://cli.github.com/))

## 🎯 Step-by-Step Deployment

### Step 1: Prepare Your Local Repository

1. **Open Terminal/Command Prompt** and navigate to your project:
```bash
cd /path/to/supplier_recommendation_app
```

2. **Verify all files are present:**
```bash
ls -la
```

You should see:
- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `data/` folder with Excel files
- `utils/` folder with Python modules

### Step 2: Initialize Git Repository

```bash
# Initialize git
git init

# Check status
git status
```

### Step 3: Configure Git (First Time Only)

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify
git config --list
```

### Step 4: Create GitHub Repository

#### Option A: Using GitHub CLI (Recommended)

```bash
# Login to GitHub
gh auth login

# Create repository
gh repo create supplier-recommendation-system --public --source=. --remote=origin

# Description
gh repo edit --description "AI-powered supplier recommendation system with ML clustering and analytics"
```

#### Option B: Using GitHub Website

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `supplier-recommendation-system`
3. Description: "AI-powered supplier recommendation system with ML clustering and analytics"
4. Choose Public or Private
5. **DO NOT** initialize with README (you already have one)
6. Click "Create repository"
7. Copy the repository URL (e.g., `https://github.com/YOUR_USERNAME/supplier-recommendation-system.git`)

### Step 5: Add Files to Git

```bash
# Add all files
git add .

# Check what will be committed
git status

# Commit
git commit -m "Initial commit: Smart Supplier Recommendation System with ML models"
```

### Step 6: Push to GitHub

#### If you used GitHub CLI:
```bash
git push -u origin main
```

#### If you created repo on website:
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/supplier-recommendation-system.git

# Rename branch to main
git branch -M main

# Push
git push -u origin main
```

### Step 7: Verify on GitHub

1. Go to `https://github.com/YOUR_USERNAME/supplier-recommendation-system`
2. You should see all your files
3. README.md should be displayed automatically

## 🌐 Deploy to Streamlit Cloud

### Step 1: Sign Up for Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "Sign up" or "Get started"
3. Choose "Continue with GitHub"
4. Authorize Streamlit to access your GitHub account

### Step 2: Create New App

1. Click "New app" button
2. Select your repository: `YOUR_USERNAME/supplier-recommendation-system`
3. Branch: `main`
4. Main file path: `app.py`
5. (Optional) Custom subdomain: `your-supplier-app`

### Step 3: Configure App Settings

1. Click "Advanced settings" (optional)
2. **Python version:** 3.11
3. **Resource limits:** Default (can upgrade later if needed)

### Step 4: Handle Data Files

⚠️ **Important:** Excel files are large and may exceed GitHub's file size limits.

#### Solution 1: Use Git LFS (For files < 100MB each)

```bash
# Install Git LFS
git lfs install

# Track Excel files
git lfs track "data/*.xlsx"

# Add .gitattributes
git add .gitattributes

# Commit
git commit -m "Add Git LFS tracking for data files"

# Push
git push
```

#### Solution 2: Use Cloud Storage (Recommended for large files)

1. **Upload files to Google Drive:**
   - Upload all Excel files to a folder
   - Right-click each file → "Get link" → "Anyone with the link can view"
   - Copy the file ID from the URL: `https://drive.google.com/file/d/FILE_ID/view`

2. **Update `data_loader.py` to download files:**

```python
import gdown
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Google Drive file IDs
        self.file_ids = {
            'SupplierList.xlsx': 'YOUR_FILE_ID_1',
            'SupplierFinancialHealth.xlsx': 'YOUR_FILE_ID_2',
            # ... add all files
        }
        
        self._download_data_files()
    
    def _download_data_files(self):
        """Download data files from Google Drive if not present"""
        for filename, file_id in self.file_ids.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, str(file_path), quiet=False)
```

3. **Add gdown to requirements.txt:**
```bash
echo "gdown>=4.7.0" >> requirements.txt
```

4. **Commit and push:**
```bash
git add .
git commit -m "Add cloud storage support for data files"
git push
```

#### Solution 3: Use Streamlit Secrets

1. In Streamlit Cloud, go to your app
2. Click "⚙️ Settings" → "Secrets"
3. Add file URLs in TOML format:

```toml
[data_files]
supplier_list = "https://your-storage-url/SupplierList.xlsx"
financial_health = "https://your-storage-url/SupplierFinancialHealth.xlsx"
performance = "https://your-storage-url/SupplierPerformance.xlsx"
risk = "https://your-storage-url/SupplierRisk.xlsx"
sustainability = "https://your-storage-url/SupplierSustainbility.xlsx"
pdt = "https://your-storage-url/PurchasingDeliveryTool.xlsx"
delivery = "https://your-storage-url/SupplierDeliveryStatus.xlsx"
commercial = "https://your-storage-url/SupplierCommercialData.xlsx"
materials = "https://your-storage-url/MaterialCategory.xlsx"
country_risk = "https://your-storage-url/CountryRisk.xlsx"
agreements = "https://your-storage-url/AgreementList.xlsx"
entities = "https://your-storage-url/EnitityInformation.xlsx"
```

4. Update `data_loader.py`:
```python
import streamlit as st
import requests

def download_from_secrets(self):
    """Download files from URLs in Streamlit secrets"""
    if 'data_files' in st.secrets:
        for key, url in st.secrets['data_files'].items():
            filename = key.replace('_', '') + '.xlsx'
            file_path = self.data_dir / filename
            if not file_path.exists():
                response = requests.get(url)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
```

### Step 5: Deploy!

1. Click "Deploy" button
2. Wait for deployment (usually 2-5 minutes)
3. Watch the logs for any errors
4. Once complete, your app will be live!

### Step 6: Get Your App URL

Your app will be available at:
```
https://YOUR_USERNAME-supplier-recommendation-system.streamlit.app
```

Or with custom subdomain:
```
https://your-supplier-app.streamlit.app
```

## 🔄 Updating Your App

Whenever you make changes:

```bash
# Make your changes to the code

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add new feature: advanced filtering"

# Push to GitHub
git push

# Streamlit Cloud will automatically redeploy!
```

## 🐛 Troubleshooting

### Issue: "File not found" error in Streamlit Cloud

**Cause:** Data files not uploaded or path incorrect

**Solution:**
1. Check file paths in `data_loader.py`
2. Ensure files are in the `data/` directory
3. Use one of the cloud storage solutions above

### Issue: "Memory limit exceeded"

**Cause:** Too much data loaded at once

**Solution:**
1. Implement data sampling in `data_loader.py`:
```python
df = pd.read_excel(file_path).sample(frac=0.5)  # Use 50% of data
```

2. Or upgrade to Streamlit Cloud Pro for more resources

### Issue: "Module not found" error

**Cause:** Missing dependency in requirements.txt

**Solution:**
```bash
# Add missing package
echo "package-name>=version" >> requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Add missing dependency"
git push
```

### Issue: App is slow

**Solution:**
1. Ensure caching is enabled (already done in `app.py`)
2. Reduce data size
3. Optimize model training
4. Consider upgrading to Streamlit Cloud Pro

### Issue: Git push rejected (file too large)

**Cause:** File exceeds GitHub's 100MB limit

**Solution:**
1. Use Git LFS (see Solution 1 above)
2. Or use cloud storage (see Solution 2 above)
3. Or add file to `.gitignore`:
```bash
echo "data/LargeFile.xlsx" >> .gitignore
git add .gitignore
git commit -m "Ignore large file"
git push
```

## 📊 Monitoring Your App

### View Logs

1. In Streamlit Cloud, go to your app
2. Click "⋮" menu → "Logs"
3. Monitor real-time logs for errors

### Analytics

1. Streamlit Cloud provides basic analytics:
   - Number of views
   - Active users
   - Resource usage

2. For advanced analytics, integrate Google Analytics:
```python
# Add to app.py
import streamlit.components.v1 as components

# Google Analytics tracking code
ga_code = """
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
"""
components.html(ga_code, height=0)
```

## 🔒 Security Best Practices

1. **Never commit sensitive data:**
   - API keys
   - Database passwords
   - Confidential supplier information

2. **Use Streamlit Secrets for sensitive data:**
   - Store in Streamlit Cloud → Settings → Secrets
   - Access via `st.secrets`

3. **Use environment variables:**
```python
import os
api_key = os.environ.get('API_KEY', st.secrets.get('api_key'))
```

4. **Make repository private if needed:**
```bash
gh repo edit --visibility private
```

## 🎉 Success Checklist

- [ ] Repository created on GitHub
- [ ] All code pushed successfully
- [ ] Data files handled (LFS or cloud storage)
- [ ] App deployed to Streamlit Cloud
- [ ] App loads without errors
- [ ] All features working correctly
- [ ] README.md displays properly
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up

## 📞 Getting Help

- **Streamlit Community Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **Streamlit Documentation:** [docs.streamlit.io](https://docs.streamlit.io)
- **GitHub Issues:** Create an issue in your repository
- **Stack Overflow:** Tag questions with `streamlit` and `python`

## 🎓 Next Steps

1. **Add authentication:**
   - Use `streamlit-authenticator` package
   - Implement user login/logout

2. **Add database integration:**
   - Connect to PostgreSQL/MySQL
   - Store user preferences and history

3. **Implement real-time updates:**
   - Auto-refresh data from APIs
   - Scheduled data updates

4. **Add email notifications:**
   - Send recommendation reports
   - Alert on high-risk suppliers

5. **Create API endpoints:**
   - Use FastAPI alongside Streamlit
   - Expose recommendation API

---

**Congratulations! Your Smart Supplier Recommendation System is now live! 🎉**

Share your app URL with stakeholders and start getting intelligent supplier recommendations!
