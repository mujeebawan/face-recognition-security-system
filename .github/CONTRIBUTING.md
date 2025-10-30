# Contributing to Face Recognition Security System

## Important Notice

**This is proprietary software.** The project is currently in active development and **not open for public contributions** at this time.

## For Team Members

If you are an authorized team member working on this project:

### 1. Development Workflow

1. Create a feature branch from `master`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code standards below

3. Test thoroughly using the test suite
   ```bash
   pytest tests/ -v
   ```

4. Commit with clear, descriptive messages
   ```bash
   git commit -m "Add feature: Description of what you added"
   ```

5. Push and create Pull Request for review
   ```bash
   git push origin feature/your-feature-name
   ```

### 2. Code Standards

**Python Code Style:**
- Follow PEP 8 guidelines
- Use Black formatter (line length: 100)
- Use type hints for function parameters and return values
- Write docstrings for all public functions

**Code Formatting:**
```bash
# Format code
black app/ tests/

# Check style
flake8 app/ tests/

# Type checking
mypy app/
```

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `FaceRecognizer`)
- Functions/methods: `snake_case` (e.g., `detect_faces`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_FACES_PER_FRAME`)
- Private methods: `_leading_underscore` (e.g., `_process_internal`)

### 3. Commit Message Format

Use conventional commit format:

```
type(scope): brief description

Detailed explanation of what changed and why.

- Bullet points for specific changes
- Reference issue numbers if applicable (#123)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting (no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, tooling

**Examples:**
```
feat(recognition): Add multi-image enrollment support

- Implemented batch processing for multiple images
- Added validation for image count (1-10)
- Updated API endpoint to handle file arrays

Closes #45
```

```
fix(detection): Resolve GPU memory leak in SCRFD detector

The detector was not properly releasing CUDA tensors
after each frame. Added explicit cleanup in finally block.
```

### 4. Pull Request Guidelines

**PR Title:** Should be clear and descriptive

**PR Description Must Include:**
- What: Brief summary of changes
- Why: Reason for the change
- How: Technical approach used
- Testing: What tests were added/run
- Screenshots: If UI changes involved

**PR Template:**
```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing Done
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] GPU performance verified (if applicable)

## Related Issues
Closes #123

## Screenshots (if applicable)
[Add screenshots here]
```

### 5. Testing Requirements

All code changes must include appropriate tests:

**Required Tests:**
- Unit tests for new functions
- Integration tests for API endpoints
- Performance tests for GPU operations

**Running Tests:**
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/integration/test_recognition.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

**Test File Naming:**
- `test_*.py` for test files
- `test_<module>_<function>` for test functions

### 6. Documentation Requirements

Update documentation when:
- Adding new features
- Changing API endpoints
- Modifying configuration options
- Updating dependencies

**Files to Update:**
- README.md (if user-facing changes)
- API documentation (docs/api/)
- Code comments (for complex logic)
- ARCHITECTURE.md (for architectural changes)

### 7. Security Guidelines

**Never Commit:**
- API keys, passwords, or credentials
- `.env` files with real values
- Database files
- Camera IP addresses or credentials
- Personal information

**Security Checklist:**
- [ ] No hardcoded credentials
- [ ] Input validation implemented
- [ ] SQL injection prevention (use parameterized queries)
- [ ] File upload validation (type, size checks)
- [ ] Error messages don't leak sensitive info

### 8. Performance Guidelines

For GPU-related code:
- Always profile before optimizing
- Use TensorRT where possible
- Implement proper CUDA memory management
- Test on actual Jetson hardware

**Performance Testing:**
```bash
python3 scripts/utilities/benchmark_performance.py
```

### 9. Code Review Process

All code must be reviewed before merging:

**Reviewer Checklist:**
- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security issues
- [ ] Performance impact considered
- [ ] No breaking changes (or properly documented)

**Review Response Time:**
- Critical fixes: 24 hours
- Features: 48-72 hours
- Documentation: 48 hours

### 10. Branch Naming

Use descriptive branch names:

```
feature/add-multi-angle-capture
fix/camera-connection-timeout
docs/update-api-reference
refactor/detector-memory-management
perf/optimize-embedding-search
```

### 11. Dependencies

When adding new dependencies:

1. Check license compatibility
2. Verify Jetson ARM64 compatibility
3. Test with Python 3.10
4. Document in requirements.txt
5. Update LICENSE with attribution

**Adding Dependency:**
```bash
pip install <package>
pip freeze | grep <package> >> requirements.txt
```

### 12. GitHub Project Board

This project uses GitHub Projects for task management:
- https://github.com/users/mujeebawan/projects/5

**Workflow:**
1. Issues are automatically added to board
2. Move to "In Progress" when starting work
3. Link PR to issue
4. Move to "Done" when merged

**Issue Labels:**
- `bug`: Something isn't working
- `feature`: New functionality
- `enhancement`: Improvement to existing feature
- `documentation`: Documentation updates
- `performance`: Performance related
- `security`: Security related
- `priority:high`: Urgent issue
- `priority:medium`: Normal priority
- `priority:low`: Nice to have

### 13. Contact

For questions or clarification:
- **Project Lead**: Muhammad Mujeeb Awan
- **Email**: mujeebciit72@gmail.com
- **GitHub Discussions**: Use for technical questions
- **Slack/Teams**: [Team communication channel if available]

---

## For External Parties

If you are interested in:
- **Commercial licensing**: Contact mujeebciit72@gmail.com
- **Research collaboration**: Contact via institutional email
- **Bug reports**: Open an issue (security issues via email)
- **Feature requests**: Open an issue for discussion

**Note**: Pull requests from non-team members will not be accepted without prior authorization.

---

*Last Updated: October 30, 2025*
*Version: 1.0*
