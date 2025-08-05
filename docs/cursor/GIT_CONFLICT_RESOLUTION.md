# Git Conflict Resolution Guide

## 🎯 **Overview**

This document provides comprehensive guidance on handling Git conflicts, especially during significant structural changes like reorganizing project directories, moving files, and refactoring code.

## 🚨 **Common Conflict Scenarios**

### 1. **Structural Changes (Most Common)**
- Moving files to new directories
- Renaming files and directories
- Reorganizing project structure
- Adding new directories with `__init__.py` files

### 2. **Code Changes**
- Modifying existing files
- Adding new features
- Updating dependencies
- Changing configuration files

### 3. **Documentation Changes**
- Updating README files
- Adding new documentation
- Moving documentation to different folders

## 🔧 **Prevention Strategies**

### **Before Making Changes**

1. **Check Current Status**
   ```bash
   git status
   git log --oneline -5
   ```

2. **Fetch Latest Changes**
   ```bash
   git fetch origin
   git pull origin main
   ```

3. **Create Feature Branch (Recommended)**
   ```bash
   git checkout -b feature/structural-improvements
   ```

### **During Structural Changes**

1. **Use Git Move Instead of Delete/Add**
   ```bash
   # ✅ Good: Git tracks the move
   git mv old_file.py new_directory/old_file.py
   
   # ❌ Bad: Git sees as delete + add
   rm old_file.py
   cp old_file.py new_directory/old_file.py
   ```

2. **Stage Changes Incrementally**
   ```bash
   # Stage moves first
   git add -A
   
   # Then stage modifications
   git add modified_file.py
   ```

3. **Commit Frequently**
   ```bash
   git commit -m "refactor: move setup files to setup/cuda/"
   git commit -m "refactor: reorganize test structure"
   ```

## 🛠️ **Conflict Resolution Commands**

### **Basic Conflict Resolution**

1. **Check Conflict Status**
   ```bash
   git status
   ```

2. **See Conflict Files**
   ```bash
   git diff --name-only --diff-filter=U
   ```

3. **Resolve Conflicts**
   ```bash
   # Edit conflicted files manually
   # Look for <<<<<<< HEAD, =======, >>>>>>> markers
   
   # After resolving, stage files
   git add resolved_file.py
   ```

4. **Complete Resolution**
   ```bash
   git commit -m "resolve: merge conflicts in main.py"
   ```

### **Advanced Conflict Resolution**

#### **For Structural Changes**

1. **Use Git's Move Detection**
   ```bash
   # Git automatically detects moves when you use git add -A
   git add -A
   git status  # Should show "renamed" instead of "deleted" + "new file"
   ```

2. **Force Git to Recognize Moves**
   ```bash
   # If Git doesn't detect moves automatically
   git add -A --renames
   ```

3. **Manual Move Tracking**
   ```bash
   # For complex moves, track them manually
   git mv old_path/file.py new_path/file.py
   git commit -m "refactor: move file to new location"
   ```

#### **For Large Structural Changes**

1. **Create Backup Branch**
   ```bash
   git checkout -b backup/before-restructure
   git checkout main
   ```

2. **Make Changes Incrementally**
   ```bash
   # Step 1: Create new directories
   mkdir setup/cuda
   git add setup/
   git commit -m "feat: create setup directory structure"
   
   # Step 2: Move files
   git mv install_cuda.py setup/cuda/
   git commit -m "refactor: move CUDA installation files"
   
   # Step 3: Update references
   # Edit files that reference moved files
   git add .
   git commit -m "refactor: update file references"
   ```

## 📋 **Step-by-Step Resolution Process**

### **Scenario: Major Project Restructure**

#### **Step 1: Prepare**
```bash
# Check current status
git status
git fetch origin

# Create feature branch
git checkout -b feature/project-restructure
```

#### **Step 2: Make Changes**
```bash
# Create new directory structure
mkdir -p setup/cuda tests/unit tests/integration tests/diagnostic

# Move files (Git will track moves)
git mv install_cuda.py setup/cuda/
git mv test_*.py tests/unit/
git mv check_*.py tests/diagnostic/

# Add new files
git add setup/ tests/
```

#### **Step 3: Commit Changes**
```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "refactor: major project structure improvements

- Reorganize test structure with categorized directories
- Move CUDA setup files to setup/cuda/ directory
- Update file references and documentation
- Improve maintainability with proper structure"
```

#### **Step 4: Push Changes**
```bash
# Push to remote
git push origin feature/project-restructure

# Or if working on main branch
git push origin main
```

## 🚨 **Emergency Conflict Resolution**

### **When Conflicts Occur**

1. **Stop and Assess**
   ```bash
   git status
   git log --oneline -10
   ```

2. **Abort if Necessary**
   ```bash
   # If you need to start over
   git merge --abort
   git rebase --abort
   ```

3. **Use Stash for Temporary Work**
   ```bash
   # Save current work
   git stash push -m "work in progress"
   
   # Pull latest changes
   git pull origin main
   
   # Restore work
   git stash pop
   ```

### **Force Push (Use with Caution)**

```bash
# Only use when you're sure about your changes
git push --force-with-lease origin main
```

## 📊 **Conflict Prevention Checklist**

### **Before Starting Work**
- [ ] `git status` - Check current state
- [ ] `git fetch origin` - Get latest remote changes
- [ ] `git pull origin main` - Update local branch
- [ ] Create feature branch if making significant changes

### **During Work**
- [ ] Use `git mv` instead of delete/add for moves
- [ ] Stage changes incrementally
- [ ] Commit frequently with descriptive messages
- [ ] Test changes before committing

### **Before Pushing**
- [ ] `git status` - Ensure clean working directory
- [ ] `git log --oneline -5` - Review recent commits
- [ ] `git diff origin/main` - Check what will be pushed
- [ ] Consider using `--force-with-lease` for structural changes

## 🔍 **Troubleshooting Common Issues**

### **Issue: Git Not Detecting File Moves**

**Solution:**
```bash
# Force Git to detect moves
git add -A --renames

# Or manually track moves
git mv old_file.py new_directory/old_file.py
```

### **Issue: Merge Conflicts in Multiple Files**

**Solution:**
```bash
# Resolve conflicts one by one
git status  # See conflicted files
git add resolved_file1.py
git add resolved_file2.py
git commit -m "resolve: merge conflicts"
```

### **Issue: Remote Has Changes You Don't Want**

**Solution:**
```bash
# Create backup of your changes
git stash push -m "my changes"

# Pull remote changes
git pull origin main

# Apply your changes on top
git stash pop
```

### **Issue: Accidentally Committed to Wrong Branch**

**Solution:**
```bash
# Create new branch with your changes
git checkout -b feature/my-changes

# Reset main branch
git checkout main
git reset --hard origin/main

# Switch back to your feature branch
git checkout feature/my-changes
```

## 📚 **Best Practices**

### **For Structural Changes**

1. **Plan Your Changes**
   - Document the new structure before making changes
   - Create a checklist of files to move
   - Identify files that need reference updates

2. **Use Descriptive Commit Messages**
   ```bash
   git commit -m "refactor: reorganize project structure

   - Move setup files to setup/cuda/
   - Reorganize tests into unit/, integration/, diagnostic/
   - Update import statements and documentation
   - Improve maintainability and organization"
   ```

3. **Test After Each Major Change**
   ```bash
   # Run tests to ensure nothing is broken
   python run_tests.py
   
   # Check if imports still work
   python -c "import modules.config_manager"
   ```

### **For Team Collaboration**

1. **Communicate Changes**
   - Notify team members about structural changes
   - Document the new structure
   - Provide migration guide if needed

2. **Use Feature Branches**
   ```bash
   git checkout -b feature/new-structure
   # Make changes
   git push origin feature/new-structure
   # Create pull request
   ```

3. **Review Before Merging**
   - Have team members review structural changes
   - Test the new structure thoroughly
   - Update documentation

## 🎯 **Summary**

### **Key Takeaways**

1. **Prevention is Better**: Use proper Git commands to avoid conflicts
2. **Plan Your Changes**: Document structure changes before implementing
3. **Commit Frequently**: Small, incremental commits are easier to manage
4. **Use Feature Branches**: Isolate changes from main branch
5. **Test Thoroughly**: Ensure changes don't break existing functionality

### **Quick Reference**

```bash
# Before making changes
git status && git fetch origin && git pull origin main

# During structural changes
git mv old_file.py new_directory/old_file.py
git add -A
git commit -m "refactor: descriptive message"

# After changes
git push origin main
```

### **Emergency Commands**

```bash
# Abort current operation
git merge --abort
git rebase --abort

# Reset to last good state
git reset --hard HEAD~1

# Stash work in progress
git stash push -m "emergency stash"
```

This guide should help you handle Git conflicts effectively, especially during significant structural changes like the ones we just completed! 🚀 