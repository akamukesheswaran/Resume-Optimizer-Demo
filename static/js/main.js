// ============================================================================
// RESUME MATCHER - MAIN JAVASCRIPT
// ============================================================================

// Global variables
let currentAnalysisData = null;

// ============================================================================
// FILE UPLOAD HANDLING
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('resume_file');
    const fileName = document.getElementById('file-name');
    const resumeText = document.getElementById('resume_text');
    
    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileName.textContent = file.name;
            fileName.style.color = '#2563eb';
            
            // Clear text area if file is selected
            resumeText.value = '';
            resumeText.disabled = true;
            resumeText.style.opacity = '0.5';
        } else {
            fileName.textContent = 'Choose File (.pdf, .docx, .txt)';
            fileName.style.color = '';
            resumeText.disabled = false;
            resumeText.style.opacity = '1';
        }
    });
    
    // Allow re-enabling text area if file is cleared
    resumeText.addEventListener('focus', function() {
        if (fileInput.files.length > 0) {
            if (confirm('Clear file selection and use text input instead?')) {
                fileInput.value = '';
                fileName.textContent = 'Choose File (.pdf, .docx, .txt)';
                fileName.style.color = '';
                resumeText.disabled = false;
                resumeText.style.opacity = '1';
            }
        }
    });
});

// ============================================================================
// FORM SUBMISSION & ANALYSIS
// ============================================================================

document.getElementById('analyzeForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    const errorDisplay = document.getElementById('errorDisplay');
    const resultsSection = document.getElementById('resultsSection');
    
    // Validate input
    const resumeText = document.getElementById('resume_text').value.trim();
    const resumeFile = document.getElementById('resume_file').files[0];
    const jobDescription = document.getElementById('job_description').value.trim();
    
    if (!jobDescription) {
        showError('Please provide a job description');
        return;
    }
    
    if (!resumeText && !resumeFile) {
        showError('Please provide your resume (either text or file)');
        return;
    }
    
    // Hide previous results and errors
    resultsSection.style.display = 'none';
    errorDisplay.style.display = 'none';
    
    // Show loading state
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('job_description', jobDescription);
        
        if (resumeFile) {
            formData.append('resume_file', resumeFile);
        } else {
            formData.append('resume_text', resumeText);
        }
        
        // Send request
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Store data for later use
        currentAnalysisData = data;
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Update overall score
    const scoreValue = document.getElementById('scoreValue');
    const scoreLabel = document.getElementById('scoreLabel');
    const scoreCircle = document.getElementById('scoreCircle');
    
    const scorePercent = Math.round(data.final_score * 100);
    scoreValue.textContent = `${scorePercent}%`;
    scoreLabel.textContent = data.recommendation;
    
    // Set color based on score
    scoreCircle.classList.remove('strong', 'good', 'weak');
    if (data.final_score >= 0.80) {
        scoreCircle.classList.add('strong');
    } else if (data.final_score >= 0.60) {
        scoreCircle.classList.add('good');
    } else {
        scoreCircle.classList.add('weak');
    }
    
    // Update breakdown
    updateProgressBar('bert', data.bert_similarity);
    updateProgressBar('tfidf', data.tfidf_similarity);
    updateProgressBar('category', data.category_match);
    updateProgressBar('skills', data.skill_score);
    
    // Update category info
    document.getElementById('categoryInfo').textContent = 
        `Resume: ${data.resume_category} | Job: ${data.job_category}`;
    
    // Update skills
    displaySkills(data.matched_skills, data.missing_skills);
    
    // Show optimization section if needed
    const optimizationSection = document.getElementById('optimizationSection');
    if (data.needs_optimization) {
        optimizationSection.style.display = 'block';
        // Reset suggestions display
        document.getElementById('suggestionsDisplay').style.display = 'none';
        document.getElementById('suggestionsContent').textContent = '';
        document.getElementById('approvedPoints').value = '';
    } else {
        optimizationSection.style.display = 'none';
    }
}

function updateProgressBar(id, value) {
    const progress = document.getElementById(`${id}Progress`);
    const valueDisplay = document.getElementById(`${id}Value`);
    
    const percent = Math.round(value * 100);
    progress.style.width = `${percent}%`;
    valueDisplay.textContent = `${percent}%`;
}

function displaySkills(matched, missing) {
    const matchedSkills = document.getElementById('matchedSkills');
    const missingSkills = document.getElementById('missingSkills');
    const matchedCount = document.getElementById('matchedCount');
    const missingCount = document.getElementById('missingCount');
    
    // Clear existing content
    matchedSkills.innerHTML = '';
    missingSkills.innerHTML = '';
    
    // Update counts
    matchedCount.textContent = matched.length;
    missingCount.textContent = missing.length;
    
    // Display matched skills
    if (matched.length > 0) {
        matched.forEach(skill => {
            const tag = document.createElement('span');
            tag.className = 'skill-tag matched';
            tag.textContent = skill.resume_has;
            tag.title = `Matches: ${skill.job_requires} (${Math.round(skill.similarity)}% similarity)`;
            matchedSkills.appendChild(tag);
        });
    } else {
        matchedSkills.innerHTML = '<p style="color: #64748b; padding: 10px;">No matched skills found</p>';
    }
    
    // Display missing skills
    if (missing.length > 0) {
        missing.forEach(skill => {
            const tag = document.createElement('span');
            tag.className = 'skill-tag missing';
            tag.textContent = skill;
            missingSkills.appendChild(tag);
        });
    } else {
        missingSkills.innerHTML = '<p style="color: #64748b; padding: 10px;">All required skills are present!</p>';
    }
}

// ============================================================================
// AI SUGGESTIONS
// ============================================================================

document.getElementById('getSuggestionsBtn').addEventListener('click', async function() {
    const btn = this;
    const suggestionsDisplay = document.getElementById('suggestionsDisplay');
    const suggestionsContent = document.getElementById('suggestionsContent');
    
    // Show loading state
    btn.disabled = true;
    btn.textContent = 'Getting AI Suggestions...';
    suggestionsContent.textContent = 'Analyzing your resume and generating personalized suggestions... This may take 10-20 seconds.';
    suggestionsDisplay.style.display = 'block';
    
    try {
        const response = await fetch('/get_suggestions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                resume_text: currentAnalysisData.resume_text,
                job_description: currentAnalysisData.job_description,
                missing_skills: currentAnalysisData.missing_skills,
                matched_skills: currentAnalysisData.matched_skills,
                score_details: {
                    final_score: currentAnalysisData.final_score,
                    bert_similarity: currentAnalysisData.bert_similarity,
                    tfidf_similarity: currentAnalysisData.tfidf_similarity,
                    resume_category: currentAnalysisData.resume_category,
                    job_category: currentAnalysisData.job_category
                }
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display suggestions
        suggestionsContent.textContent = data.suggestions;
        
        // Scroll to suggestions
        suggestionsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        suggestionsContent.textContent = `Error: ${error.message}`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Get AI Suggestions';
    }
});

// ============================================================================
// OPTIMIZE & DOWNLOAD RESUME
// ============================================================================

document.getElementById('optimizeResumeBtn').addEventListener('click', async function() {
    const btn = this;
    const approvedPoints = document.getElementById('approvedPoints').value.trim();
    const suggestionsContent = document.getElementById('suggestionsContent').textContent;
    
    if (!approvedPoints) {
        alert('Please add your approved changes or specific instructions in the text area above.');
        return;
    }
    
    // Show loading state
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.innerHTML = '<span class="loader"></span> Optimizing Resume...';
    
    try {
        const response = await fetch('/optimize_resume', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                resume_text: currentAnalysisData.resume_text,
                job_description: currentAnalysisData.job_description,
                suggestions: suggestionsContent,
                approved_points: approvedPoints
            })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to optimize resume');
        }
        
        // Download the file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'optimized_resume.docx';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        // Show success message
        btn.textContent = '✅ Downloaded!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 3000);
        
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    const errorDisplay = document.getElementById('errorDisplay');
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';
    errorDisplay.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorDisplay.style.display = 'none';
    }, 5000);
}