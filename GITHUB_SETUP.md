# 🚀 GitHub Setup - Quick Guide

Follow these steps to push your Supplier Recommendation System to GitHub.

## Option 1: Using GitHub CLI (Easiest)

### Step 1: Install GitHub CLI

**macOS:**
```bash
brew install gh
```

**Windows:**
Download from [cli.github.com](https://cli.github.com/)

**Linux:**
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### Step 2: Login to GitHub

```bash
gh auth login
```

Follow the prompts:
1. Choose "GitHub.com"
2. Choose "HTTPS"
3. Authenticate with your browser

### Step 3: Create and Push Repository

```bash
# Navigate to your project
cd /path/to/supplier_recommendation_app

# Initialize git
git init

# Create repository on GitHub
gh repo create supplier-recommendation-system --public --source=. --remote=origin

# Add all files
git add .

# Commit
git commit -m "Initial commit: Smart Supplier Recommendation System"

# Push
git push -u origin main
```

### Step 4: Verify

```bash
gh repo view --web
```

This will open your repository in the browser!

---

## Option 2: Using Git Command Line

### Step 1: Create Repository on GitHub Website

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `supplier-recommendation-system`
3. Description: "AI-powered supplier recommendation system"
4. Choose Public or Private
5. **DO NOT** check "Initialize with README"
6. Click "Create repository"

### Step 2: Initialize Local Repository

```bash
# Navigate to your project
cd /path/to/supplier_recommendation_app

# Initialize git
git init

# Configure git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Add all files
git add .

# Commit
git commit -m "Initial commit: Smart Supplier Recommendation System"
```

### Step 3: Connect to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/supplier-recommendation-system.git

# Rename branch to main
git branch -M main

# Push
git push -u origin main
```

### Step 4: Enter Credentials

When prompted:
- **Username:** Your GitHub username
- **Password:** Your Personal Access Token (NOT your GitHub password)

#### How to Create Personal Access Token:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: "Supplier Recommendation System"
4. Expiration: Choose duration
5. Scopes: Check `repo` (full control of private repositories)
6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again!)
8. Use this token as your password when pushing

---

## Handling Data Files

⚠️ **Important:** Excel files may be too large for GitHub (limit: 100MB per file)

### Option A: Use Git LFS (for files < 100MB)

```bash
# Install Git LFS
git lfs install

# Track Excel files
git lfs track "data/*.xlsx"

# Add .gitattributes
git add .gitattributes

# Commit
git commit -m "Add Git LFS for data files"

# Push
git push
```

### Option B: Ignore Data Files (Recommended)

```bash
# Add data files to .gitignore
echo "data/*.xlsx" >> .gitignore

# Commit
git add .gitignore
git commit -m "Ignore data files"
git push
```

Then use cloud storage (see DEPLOYMENT_GUIDE.md) for data files.

---

## Updating Your Repository

After making changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## Common Issues & Solutions

### Issue: "Permission denied (publickey)"

**Solution:** Use HTTPS instead of SSH, or set up SSH keys:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

### Issue: "fatal: remote origin already exists"

**Solution:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/supplier-recommendation-system.git
```

### Issue: "failed to push some refs"

**Solution:**
```bash
# Pull first
git pull origin main --rebase

# Then push
git push origin main
```

### Issue: "File too large" (>100MB)

**Solution:** Use Git LFS or move to cloud storage (see above)

---

## Next Steps

After pushing to GitHub:

1. ✅ Verify files on GitHub
2. ✅ Check README displays correctly
3. ✅ Deploy to Streamlit Cloud (see DEPLOYMENT_GUIDE.md)
4. ✅ Share your repository URL!

---

## Quick Reference

```bash
# Clone your repo (on another machine)
git clone https://github.com/YOUR_USERNAME/supplier-recommendation-system.git

# Check status
git status

# View commit history
git log --oneline

# Create branch
git checkout -b feature-name

# Switch branch
git checkout main

# Merge branch
git merge feature-name

# Delete branch
git branch -d feature-name

# View remotes
git remote -v

# Pull latest changes
git pull

# Push to specific branch
git push origin branch-name
```

---

**Need Help?**

- GitHub Docs: [docs.github.com](https://docs.github.com)
- GitHub CLI Docs: [cli.github.com/manual](https://cli.github.com/manual)
- Git Cheat Sheet: [education.github.com/git-cheat-sheet-education.pdf](https://education.github.com/git-cheat-sheet-education.pdf)

---

**Ready to deploy? See DEPLOYMENT_GUIDE.md for Streamlit Cloud deployment!**
