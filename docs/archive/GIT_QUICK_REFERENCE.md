# Git Quick Reference for Structural Changes

## 🚀 **Quick Commands for Structural Changes**

### **Before Making Changes**
```bash
git status                    # Check current state
git fetch origin             # Get latest remote changes
git pull origin main         # Update local branch
```

### **During Structural Changes**
```bash
# Create new directories
mkdir new_directory

# Move files (Git tracks moves)
git mv old_file.py new_directory/old_file.py

# Stage all changes including moves
git add -A

# Verify moves detected
git status
```

### **After Changes**
```bash
# Commit with descriptive message
git commit -m "refactor: descriptive title

- List specific changes
- Explain benefits
- Note any breaking changes"

# Push to remote
git push origin main
```

## 🚨 **Emergency Commands**

### **Abort Current Operation**
```bash
git merge --abort
git rebase --abort
```

### **Reset to Last Good State**
```bash
git reset --hard HEAD~1
```

### **Stash Work in Progress**
```bash
git stash push -m "emergency stash"
git stash pop
```

### **Force Push (Use with Caution)**
```bash
git push --force-with-lease origin main
```

## 📋 **Conflict Prevention Checklist**

### **Before Starting**
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

## 🔧 **Common Scenarios**

### **Moving Files to New Directory**
```bash
# Create directory
mkdir new_directory

# Move file
git mv old_file.py new_directory/old_file.py

# Stage changes
git add -A

# Commit
git commit -m "refactor: move file to new directory"
```

### **Reorganizing Test Structure**
```bash
# Create test categories
mkdir tests/unit tests/integration tests/diagnostic

# Move test files
git mv test_*.py tests/unit/
git mv check_*.py tests/diagnostic/

# Stage all changes
git add -A

# Commit
git commit -m "refactor: reorganize test structure"
```

### **Setting Up New Project Structure**
```bash
# Create main directories
mkdir setup docs/cursor

# Move setup files
git mv install_*.py setup/
git mv setup_*.py setup/

# Move documentation
git mv docs/process_*.md docs/cursor/

# Stage all changes
git add -A

# Commit
git commit -m "refactor: create organized project structure"
```

## 📊 **Git Status Meanings**

### **File Status Indicators**
- `M` - Modified file
- `A` - Added file
- `D` - Deleted file
- `R` - Renamed file (Git detected move)
- `??` - Untracked file

### **Example Status Output**
```bash
git status

# Output:
# modified:   README.md
# renamed:    install_cuda.py -> setup/cuda/install_cuda.py
# new file:   setup/README.md
# deleted:    old_file.py
```

## 🎯 **Best Practices**

### **Commit Messages**
```bash
# ✅ Good
git commit -m "refactor: reorganize project structure

- Move setup files to setup/cuda/
- Reorganize tests into unit/, integration/, diagnostic/
- Update documentation and README files
- Improve maintainability and organization"

# ❌ Bad
git commit -m "fix stuff"
```

### **File Organization**
```bash
# ✅ Good: Use descriptive directory names
setup/cuda/
tests/unit/
docs/cursor/

# ❌ Bad: Vague names
stuff/
temp/
old/
```

### **Testing After Changes**
```bash
# Test imports still work
python -c "import modules.config_manager"

# Run tests
python run_tests.py

# Check if setup works
python setup/cuda/install_cuda.py
```

## 🚨 **Troubleshooting**

### **Git Not Detecting Moves**
```bash
# Force Git to detect moves
git add -A --renames

# Or manually track moves
git mv old_file.py new_directory/old_file.py
```

### **Conflicts During Merge**
```bash
# See conflicted files
git status

# Resolve conflicts manually
# Edit files with <<<<<<< HEAD markers

# After resolving
git add resolved_file.py
git commit -m "resolve: merge conflicts"
```

### **Accidentally Committed to Wrong Branch**
```bash
# Create new branch with your changes
git checkout -b feature/my-changes

# Reset main branch
git checkout main
git reset --hard origin/main

# Switch back to your feature branch
git checkout feature/my-changes
```

## 📚 **Related Documentation**

- **[GIT_CONFLICT_RESOLUTION.md](GIT_CONFLICT_RESOLUTION.md)** - Comprehensive conflict resolution guide
- **[STRUCTURAL_CHANGES_CASE_STUDY.md](STRUCTURAL_CHANGES_CASE_STUDY.md)** - Real-world case study
- **[SETUP_STRUCTURE_IMPROVEMENTS.md](SETUP_STRUCTURE_IMPROVEMENTS.md)** - Setup organization guide
- **[TEST_STRUCTURE_IMPROVEMENTS.md](TEST_STRUCTURE_IMPROVEMENTS.md)** - Test organization guide

## 🎉 **Quick Success Tips**

1. **Plan Before Acting**: Document your changes
2. **Use Git Commands Properly**: `git mv` for moves, `git add -A` for staging
3. **Commit Frequently**: Small, incremental commits
4. **Test After Changes**: Ensure nothing is broken
5. **Document Process**: Keep track in `docs/cursor/`

This quick reference should help you handle Git conflicts effectively during structural changes! 🚀 