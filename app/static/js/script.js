// app/static/js/script.js

document.addEventListener("DOMContentLoaded", () => {
  console.log("DOM Content Loaded. Initializing event listeners.");

  // Get DOM elements for Resume Upload
  const resumeFile = document.getElementById("resumeFile");
  const uploadResumeBtn = document.getElementById("uploadResumeBtn");
  const clearResumeBtn = document.getElementById("clearResumeBtn");
  const resumeStatus = document.getElementById("resumeStatus");
  const resumeResult = document.getElementById("resumeResult");
  const uploadedResumeId = document.getElementById("uploadedResumeId");
  const extractedResumeName = document.getElementById("extractedResumeName");
  const extractedResumeEmail = document.getElementById("extractedResumeEmail");
  const extractedResumePhone = document.getElementById("extractedResumePhone");
  const extractedResumeSkills = document.getElementById(
      "extractedResumeSkills"
  );
  const extractedResumeExperience = document.getElementById(
      "extractedResumeExperience"
  );
  const extractedResumeEducation = document.getElementById(
      "extractedResumeEducation"
  );
  const resumeSpinner = document.getElementById("resumeSpinner");
  const resumeTokenUsageSection = document.getElementById(
      "resumeTokenUsageSection"
  );
  const resumePromptTokens = document.getElementById("resumePromptTokens");
  const resumeCompletionTokens = document.getElementById(
      "resumeCompletionTokens"
  );
  const resumeTotalTokens = document.getElementById("resumeTotalTokens");

  // Get DOM elements for Job Description File Upload
  const jobFile = document.getElementById("jobFile");
  const uploadJobBtn = document.getElementById("uploadJobBtn");
  const clearJobBtn = document.getElementById("clearJobBtn");
  const jobStatus = document.getElementById("jobStatus");
  const jobResult = document.getElementById("jobResult");
  const uploadedJobId = document.getElementById("uploadedJobId");
  const extractedJobSkills = document.getElementById("extractedJobSkills");
  const extractedJobExperience = document.getElementById(
      "extractedJobExperience"
  );
  const extractedJobEducation = document.getElementById(
      "extractedJobEducation"
  );
  const jobSpinner = document.getElementById("jobSpinner");
  const jobTokenUsageSection = document.getElementById("jobTokenUsageSection");
  const jobPromptTokens = document.getElementById("jobPromptTokens");
  const jobCompletionTokens = document.getElementById("jobCompletionTokens");
  const jobTotalTokens = document.getElementById("jobTotalTokens");

  // Get DOM elements for Paste Job Description Text (NEW)
  const jobDescriptionText = document.getElementById("jobDescriptionText");
  const submitJobTextBtn = document.getElementById("submitJobTextBtn");
  const clearJobTextBtn = document.getElementById("clearJobTextBtn");
  const jobTextSpinner = document.getElementById("jobTextSpinner");
  const jobTextStatus = document.getElementById("jobTextStatus");
  const jobTextResult = document.getElementById("jobTextResult");
  const pastedJobId = document.getElementById("pastedJobId");
  const pastedJobSkills = document.getElementById("pastedJobSkills");
  const pastedJobExperience = document.getElementById("pastedJobExperience");
  const pastedJobEducation = document.getElementById("pastedJobEducation");
  const jobTextTokenUsageSection = document.getElementById("jobTextTokenUsageSection");
  const jobTextPromptTokens = document.getElementById("jobTextPromptTokens");
  const jobTextCompletionTokens = document.getElementById("jobTextCompletionTokens");
  const jobTextTotalTokens = document.getElementById("jobTextTotalTokens");


  // Get DOM elements for Matching
  const matchResumeIdInput = document.getElementById("matchResumeId");
  const matchJobIdInput = document.getElementById("matchJobId");
  const matchSpecificBtn = document.getElementById("matchSpecificBtn");
  const findTopResumesBtn = document.getElementById("findTopResumesBtn");
  const matchResultDiv = document.getElementById("matchResult");
  const matchResultContent = document.getElementById("matchResultContent");
  const matchSpecificSpinner = document.getElementById("matchSpecificSpinner");
  const findTopResumesSpinner = document.getElementById(
      "findTopResumesSpinner"
  );
  const matchTokenUsageSection = document.getElementById(
      "matchTokenUsageSection"
  );
  const matchPromptTokens = document.getElementById("matchPromptTokens");
  const matchCompletionTokens = document.getElementById(
      "matchCompletionTokens"
  );
  const matchTotalTokens = document.getElementById("matchTotalTokens");

  // Get DOM elements for Interview Questions
  const generateQuestionsBtn = document.getElementById("generateQuestionsBtn");
  const generateQuestionsSpinner = document.getElementById(
      "generateQuestionsSpinner"
  );
  const interviewQuestionsResultDiv = document.getElementById(
      "interviewQuestionsResult"
  );
  const interviewQuestionsContent = document.getElementById(
      "interviewQuestionsContent"
  );
  const interviewQuestionsTokenUsageSection = document.getElementById(
      "interviewQuestionsTokenUsageSection"
  );
  const interviewQuestionsPromptTokens = document.getElementById(
      "interviewQuestionsPromptTokens"
  );
  const interviewQuestionsCompletionTokens = document.getElementById(
      "interviewQuestionsCompletionTokens"
  );
  const interviewQuestionsTotalTokens = document.getElementById(
      "interviewQuestionsTotalTokens"
  );

  // Store latest uploaded IDs for convenience and dynamic placeholders
  let latestResumeId = "";
  let latestJobId = "";

  // Helper function to display messages
  function showStatus(element, message, isError = false) {
      element.textContent = message;
      element.className = "mt-4 text-sm font-medium"; // Reset classes
      if (isError) {
          element.classList.add("text-red-600");
      } else {
          element.classList.add("text-green-600");
      }
      console.log(`Status Update (${isError ? "ERROR" : "INFO"}): ${message}`);
  }
 // ----------------- Batch Resume Uploader -----------------
 async function uploadResumesBatch() {
    const files = Array.from(resumeFile.files);
    if (files.length === 0) {
      showStatus(resumeStatus, "Select at least one resume.", true);
      return;
    }

    toggleLoading(uploadResumeBtn, resumeSpinner, true);
    showStatus(resumeStatus, `Uploading ${files.length} resumes…`);

    const formData = new FormData();
    // the backend expects field name "file"
    files.forEach(f => formData.append("file", f));

    try {
      const resp = await fetch("/resumes/upload", {
        method: "POST",
        body: formData
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Upload failed");

      resumeResult.innerHTML = "";
      data.results.forEach(r => {
        const div = document.createElement("div");

        if (r.error) {
          div.innerHTML = `<strong>${r.filename}</strong>: <span class="text-red-600">${r.error}</span>`;
        } else {
          const e = r.extracted_entities || {};
          const t = r.llm_token_usage_refinement || {};
          div.innerHTML = `
            <strong>${r.filename}</strong> → ID: <code>${r.resume_id}</code><br>
            Name: ${e.name || "N/A"}<br>
            Email: ${e.email || "N/A"}<br>
            Phone: ${e.phone || "N/A"}<br>
            Skills: ${(e.skills || []).join(", ") || "N/A"}<br>
            Experience: ${(e.experience || []).join("; ") || "N/A"}<br>
            Education: ${(e.education || []).join("; ") || "N/A"}<br>
            Tokens: ${t.total_tokens ?? 0}
          `;
        }

        resumeResult.appendChild(div);
      });

      resumeResult.classList.remove("hidden");
      showStatus(resumeStatus, "Batch upload complete!");
    } catch (err) {
      showStatus(resumeStatus, `Error: ${err.message}`, true);
    } finally {
      toggleLoading(uploadResumeBtn, resumeSpinner, false);
    }
  }

  // Helper function to toggle loading state
  function toggleLoading(button, spinner, isLoading) {
      if (isLoading) {
          spinner.classList.remove("hidden");
          button.disabled = true;
          button.classList.add("opacity-50", "cursor-not-allowed");
      } else {
          spinner.classList.add("hidden");
          button.disabled = false;
          button.classList.remove("opacity-50", "cursor-not-allowed");
      }
  }

  // Helper function to update token usage display
  function updateTokenUsage(
      sectionElement,
      promptSpan,
      completionSpan,
      totalSpan,
      tokenUsage
  ) {
      if (tokenUsage) {
          promptSpan.textContent = tokenUsage.prompt_tokens || 0;
          completionSpan.textContent = tokenUsage.completion_tokens || 0;
          totalSpan.textContent = tokenUsage.total_tokens || 0;
          sectionElement.style.display = "block";
      } else {
          sectionElement.style.display = "none";
      }
  }

  // Helper function to clear resume results
  function clearResumeResults() {
      resumeFile.value = "";
      resumeStatus.textContent = "";
      resumeResult.classList.add("hidden");
      uploadedResumeId.textContent = "N/A";
      extractedResumeName.textContent = "N/A";
      extractedResumeEmail.textContent = "N/A";
      extractedResumePhone.textContent = "N/A";
      extractedResumeSkills.textContent = "N/A";
      extractedResumeExperience.textContent = "N/A";
      extractedResumeEducation.textContent = "N/A";
      latestResumeId = ""; // Clear stored ID
      matchResumeIdInput.placeholder = "Enter Resume ID"; // Reset placeholder
      matchResumeIdInput.value = "";
      interviewQuestionsResultDiv.classList.add("hidden"); // Also hide questions when clearing resume
      matchResultDiv.classList.add("hidden"); // Also hide match results
      updateTokenUsage(
          resumeTokenUsageSection,
          resumePromptTokens,
          resumeCompletionTokens,
          resumeTotalTokens,
          null
      ); // Clear token usage
  }

  // Helper function to clear job file upload results
  function clearJobResults() {
      jobFile.value = "";
      jobStatus.textContent = "";
      jobResult.classList.add("hidden");
      uploadedJobId.textContent = "N/A";
      extractedJobSkills.textContent = "N/A";
      extractedJobExperience.textContent = "N/A";
      extractedJobEducation.textContent = "N/A";
      latestJobId = ""; // Clear stored ID
      matchJobIdInput.placeholder = "Enter Job ID"; // Reset placeholder
      matchJobIdInput.value = ""; // Clear auto-filled ID
      interviewQuestionsResultDiv.classList.add("hidden"); // Also hide questions when clearing job
      matchResultDiv.classList.add("hidden"); // Also hide match results
      updateTokenUsage(
          jobTokenUsageSection,
          jobPromptTokens,
          jobCompletionTokens,
          jobTotalTokens,
          null
      ); // Clear token usage
  }

  // Helper function to clear pasted job text results (NEW)
  function clearJobTextResults() {
      jobDescriptionText.value = "";
      jobTextStatus.textContent = "";
      jobTextResult.classList.add("hidden");
      pastedJobId.textContent = "N/A";
      pastedJobSkills.textContent = "N/A";
      pastedJobExperience.textContent = "N/A";
      pastedJobEducation.textContent = "N/A";
      latestJobId = ""; // Clear stored ID (since this is also a job, it affects the same latestJobId)
      matchJobIdInput.placeholder = "Enter Job ID"; // Reset placeholder
      matchJobIdInput.value = ""; // Clear auto-filled ID
      interviewQuestionsResultDiv.classList.add("hidden"); // Also hide questions when clearing job
      matchResultDiv.classList.add("hidden"); // Also hide match results
      updateTokenUsage(
          jobTextTokenUsageSection,
          jobTextPromptTokens,
          jobTextCompletionTokens,
          jobTextTotalTokens,
          null
      ); // Clear token usage
  }


  // Helper function to upload files (for both resume and job file uploads)
  async function uploadFile(
      fileInput,
      statusElement,
      resultElement,
      idDisplayElement,
      type, // "resume" or "job"
      uploadBtn,
      spinner,
      tokenSection,
      promptTokensSpan,
      completionTokensSpan,
      totalTokensSpan
  ) {
      console.log(`Attempting to upload ${type} file.`);
      const file = fileInput.files[0];
      if (!file) {
          showStatus(statusElement, "Please select a file first.", true);
          console.warn(`No file selected for ${type} upload.`);
          return;
      }
      console.log(
          `File selected: ${file.name}, type: ${file.type}, size: ${file.size} bytes`
      );

      toggleLoading(uploadBtn, spinner, true);
      showStatus(statusElement, "Uploading and processing...", false);
      resultElement.classList.add("hidden"); // Hide previous results
      idDisplayElement.textContent = "N/A"; // Clear previous ID
      updateTokenUsage(
          tokenSection,
          promptTokensSpan,
          completionTokensSpan,
          totalTokensSpan,
          null
      ); // Clear previous token usage

      const formData = new FormData();
      formData.append("file", file);
      console.log("FormData created.");

      // Updated endpoints to reflect Blueprint prefixes
      const endpoint = type === "resume" ? "/resumes/upload" : "/jobs/upload";
      console.log(`Sending POST request to ${endpoint}`);

      try {
          const response = await fetch(endpoint, {
              method: "POST",
              body: formData,
          });
          console.log("Response received from backend:", response);

          const data = await response.json();
          console.log("Response JSON data:", data);

          if (response.ok) {
              showStatus(statusElement, data.message, false);
              idDisplayElement.textContent = data[`${type}_id`];
              resultElement.classList.remove("hidden");

              if (type === "resume") {
                  latestResumeId = data.resume_id;
                  matchResumeIdInput.placeholder = `e.g., ${latestResumeId}`; // Auto-fill placeholder for convenience
                  matchResumeIdInput.value = data.resume_id; // Also set the value directly
                  // Ensure the data structure matches what's returned by your resume_routes.py
                  extractedResumeName.textContent =
                      data.extracted_entities.name || "N/A";
                  extractedResumeEmail.textContent =
                      data.extracted_entities.email || "N/A";
                  extractedResumePhone.textContent =
                      data.extracted_entities.phone || "N/A";
                  extractedResumeSkills.textContent =
                      data.extracted_entities.skills &&
                      data.extracted_entities.skills.length > 0
                          ? data.extracted_entities.skills.join(", ")
                          : "N/A";
                 // new:
const expList = document.getElementById("extractedResumeExperience");
expList.innerHTML = "";        // clear old
const experiences = data.extracted_entities.experience || [];
if (experiences.length) {
  experiences.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    expList.appendChild(li);
  });
} else {
  // show a single “N/A” bullet
  const li = document.createElement("li");
  li.textContent = "N/A";
  expList.appendChild(li);
}

                  extractedResumeEducation.textContent =
                      data.extracted_entities.education &&
                      data.extracted_entities.education.length > 0
                          ? data.extracted_entities.education.join("; ")
                          : "N/A";

                  // Corrected: Use data.llm_token_usage_refinement for resume upload
                  updateTokenUsage(
                      tokenSection,
                      promptTokensSpan,
                      completionTokensSpan,
                      totalTokensSpan,
                      data.llm_token_usage_refinement
                  );

              } else { // type === "job" (for file upload)
                  latestJobId = data.job_id;
                  matchJobIdInput.placeholder = `e.g., ${latestJobId}`; // Auto-fill placeholder for convenience
                  matchJobIdInput.value = data.job_id; // Also set the value directly
                  // Ensure the data structure matches what's returned by your job_routes.py
                  extractedJobSkills.textContent =
                      data.extracted_entities.skills_required &&
                      data.extracted_entities.skills_required.length > 0
                          ? data.extracted_entities.skills_required.join(", ")
                          : "N/A";
                  extractedJobExperience.textContent =
                      data.extracted_entities.experience_required &&
                      data.extracted_entities.experience_required.length > 0
                          ? data.extracted_entities.experience_required.join("; ")
                          : "N/A";
                  extractedJobEducation.textContent =
                      data.extracted_entities.education && // Changed from education_required as per job_routes.py
                      data.extracted_entities.education.length > 0
                          ? data.extracted_entities.education.join("; ")
                          : "N/A";

                  // Corrected: Use data.llm_token_usage for job upload
                  updateTokenUsage(
                      tokenSection,
                      promptTokensSpan,
                      completionTokensSpan,
                      totalTokensSpan,
                      data.llm_token_usage
                  );
              }
          } else {
              showStatus(
                  statusElement,
                  `Error: ${data.error || "Unknown error"}`,
                  true
              );
              resultElement.classList.add("hidden"); // Hide results on error
              console.error("API Error Response:", data);
          }
      } catch (error) {
          showStatus(statusElement, `Network error: ${error.message}`, true);
          resultElement.classList.add("hidden"); // Hide results on error
          console.error("Fetch error during upload:", error);
      } finally {
          toggleLoading(uploadBtn, spinner, false);
      }
  }

  // Event Listeners for Upload Buttons
//   uploadResumeBtn.addEventListener("click", () => {
//       console.log("Upload Resume button clicked.");
//       uploadFile(
//           resumeFile,
//           resumeStatus,
//           resumeResult,
//           uploadedResumeId,
//           "resume",
//           uploadResumeBtn,
//           resumeSpinner,
//           resumeTokenUsageSection,
//           resumePromptTokens,
//           resumeCompletionTokens,
//           resumeTotalTokens
//       );
//   });

  clearResumeBtn.addEventListener("click", () => {
      console.log("Clear Resume button clicked.");
      clearResumeResults();
  });

  uploadJobBtn.addEventListener("click", () => {
      console.log("Upload Job button clicked.");
      uploadFile(
          jobFile,
          jobStatus,
          jobResult,
          uploadedJobId,
          "job",
          uploadJobBtn,
          jobSpinner,
          jobTokenUsageSection,
          jobPromptTokens,
          jobCompletionTokens,
          jobTotalTokens
      );
  });

  clearJobBtn.addEventListener("click", () => {
      console.log("Clear Job button clicked.");
      clearJobResults();
  });

  // --- NEW: Event Listener for Paste Job Description Text Button ---
  submitJobTextBtn.addEventListener("click", async () => {
      console.log("Process Job Text button clicked.");
      const jobText = jobDescriptionText.value.trim();

      if (!jobText) {
          showStatus(jobTextStatus, "Please enter some text for the job description.", true);
          console.warn("No text provided for job description paste.");
          return;
      }

      toggleLoading(submitJobTextBtn, jobTextSpinner, true);
      showStatus(jobTextStatus, "Processing pasted job description...", false);
      jobTextResult.classList.add("hidden"); // Hide previous results
      pastedJobId.textContent = "N/A"; // Clear previous ID
      updateTokenUsage(
          jobTextTokenUsageSection,
          jobTextPromptTokens,
          jobTextCompletionTokens,
          jobTextTotalTokens,
          null
      ); // Clear previous token usage

      console.log("Sending POST request to /jobs/upload_text");

      try {
          const response = await fetch('/jobs/upload_text', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ job_description_text: jobText })
          });
          console.log("Response received from backend (text upload):", response);

          const data = await response.json();
          console.log("Response JSON data (text upload):", data);

          if (response.ok) {
              showStatus(jobTextStatus, data.message, false);
              pastedJobId.textContent = data.job_id;
              jobTextResult.classList.remove("hidden");

              latestJobId = data.job_id; // Update latestJobId
              matchJobIdInput.placeholder = `e.g., ${latestJobId}`; // Auto-fill placeholder
              matchJobIdInput.value = data.job_id; // Also set the value directly

              // Populate the results for pasted text
              pastedJobSkills.textContent =
                  data.refined_entities.skills_required &&
                  data.refined_entities.skills_required.length > 0
                      ? data.refined_entities.skills_required.join(", ")
                      : "N/A";
              pastedJobExperience.textContent =
                  data.refined_entities.experience_required &&
                  data.refined_entities.experience_required.length > 0
                      ? data.refined_entities.experience_required.join("; ")
                      : "N/A";
              pastedJobEducation.textContent =
                  data.refined_entities.education &&
                  data.refined_entities.education.length > 0
                      ? data.refined_entities.education.join("; ")
                      : "N/A";

              updateTokenUsage(
                  jobTextTokenUsageSection,
                  jobTextPromptTokens,
                  jobTextCompletionTokens,
                  jobTextTotalTokens,
                  data.llm_token_usage
              );

          } else {
              showStatus(
                  jobTextStatus,
                  `Error: ${data.error || "Unknown error"}`,
                  true
              );
              jobTextResult.classList.add("hidden"); // Hide results on error
              console.error("API Error Response (text upload):", data);
          }
      } catch (error) {
          showStatus(jobTextStatus, `Network error: ${error.message}`, true);
          jobTextResult.classList.add("hidden"); // Hide results on error
          console.error("Fetch error during text upload:", error);
      } finally {
          toggleLoading(submitJobTextBtn, jobTextSpinner, false);
      }
  });

  // --- NEW: Event Listener for Clear Paste Job Description Text Button ---
  clearJobTextBtn.addEventListener("click", () => {
      console.log("Clear Job Text button clicked.");
      clearJobTextResults();
  });


  // Event Listener for Match Specific Button
  matchSpecificBtn.addEventListener("click", async () => {
      console.log("Match Specific button clicked.");
      const resumeId = matchResumeIdInput.value.trim();
      const jobId = matchJobIdInput.value.trim();

      matchResultDiv.classList.remove("hidden"); // Ensure match result div is visible
      matchResultContent.style.color = "inherit"; // Reset text color
      interviewQuestionsResultDiv.classList.add("hidden"); // Hide questions section if visible
      updateTokenUsage(
          matchTokenUsageSection,
          matchPromptTokens,
          matchCompletionTokens,
          matchTotalTokens,
          null
      ); // Clear previous token usage

      if (!resumeId || !jobId) {
          matchResultContent.textContent =
              "Please enter both Resume ID and Job ID.";
          matchResultContent.style.color = "red";
          console.warn("Missing Resume ID or Job ID for specific match.");
          return;
      }

      toggleLoading(matchSpecificBtn, matchSpecificSpinner, true);
      matchResultContent.textContent = "Matching specific resume to job...";
      console.log(
          `Requesting specific match for Resume ID: ${resumeId}, Job ID: ${jobId}`
      );

      try {
          // Updated endpoint to reflect Blueprint prefixes
          const response = await fetch(`/jobs/match-specific/${jobId}/${resumeId}`);
          const data = await response.json();
          console.log("Specific match response:", data);

          if (response.ok) {
              if (data.match_details) {
                  let resultHtml = `Overall Score: ${(
                      data.match_details.overall_score * 100
                  ).toFixed(2)}%\n\n`;
                  resultHtml += `Semantic Similarity: ${(
                      data.match_details.semantic_similarity * 100
                  ).toFixed(2)}%\n`;
                  resultHtml += `Skill Match Score: ${(
                      data.match_details.skill_match_score * 100
                  ).toFixed(2)}%\n`;
                  resultHtml += `Experience Match Score: ${(
                      data.match_details.experience_match_score * 100
                  ).toFixed(2)}%\n\n`;
                  resultHtml += `Matched Skills: ${
                      data.match_details.matched_skills.join(", ") || "None"
                  }\n`;

                  // Display candidate and job details that were returned with the match (if present)
                  resultHtml += `\n--- Candidate Details ---\n`;
                  resultHtml += `Name: ${data.match_details.candidate_name}\n`;
                  resultHtml += `Email: ${data.match_details.candidate_email}\n`;
                  resultHtml += `Phone: ${data.match_details.candidate_phone}\n`;
                  resultHtml += `Candidate Skills: ${
                      data.match_details.candidate_extracted_skills.join(", ") || "None"
                  }\n`;
                  resultHtml += `Candidate Experience: ${
                      data.match_details.candidate_extracted_experience.join("; ") ||
                      "None"
                  }\n`;

                  resultHtml += `\n--- Job Details ---\n`;
                  resultHtml += `Required Skills: ${
                      data.match_details.job_required_skills.join(", ") || "None"
                  }\n`;
                  resultHtml += `Required Experience: ${
                      data.match_details.job_required_experience.join("; ") || "None"
                  }`;

                  matchResultContent.textContent = resultHtml;
              } else {
                  matchResultContent.textContent = "No match details found.";
              }
              // No token usage for matching, as it's not an LLM call directly
              document.getElementById('matchTokenUsageSection').style.display = 'none';
          } else {
              matchResultContent.textContent = `Error: ${
                  data.error || "Unknown error"
              }`;
              matchResultContent.style.color = "red";
              console.error("API Error for specific match:", data);
          }
      } catch (error) {
          matchResultContent.textContent = `Network error: ${error.message}`;
          matchResultContent.style.color = "red";
          console.error("Fetch error for specific match:", error);
      } finally {
          toggleLoading(matchSpecificBtn, matchSpecificSpinner, false);
      }
  });
  uploadResumeBtn.addEventListener("click", uploadResumesBatch);
  // Event Listener for Find Top Resumes Button
  findTopResumesBtn.addEventListener("click", async () => {
      console.log("Find Top Resumes button clicked.");
      const jobId = matchJobIdInput.value.trim();
      matchResultDiv.classList.remove("hidden"); // Ensure result div is visible
      matchResultContent.style.color = "inherit"; // Reset text color
      interviewQuestionsResultDiv.classList.add("hidden"); // Hide questions section if visible
      updateTokenUsage(
          matchTokenUsageSection,
          matchPromptTokens,
          matchCompletionTokens,
          matchTotalTokens,
          null
      ); // Clear previous token usage

      if (!jobId) {
          matchResultContent.textContent =
              "Please enter a Job ID to find top resumes.";
          matchResultContent.style.color = "red";
          console.warn("Missing Job ID for top resumes search.");
          return;
      }

      toggleLoading(findTopResumesBtn, findTopResumesSpinner, true);
      matchResultContent.textContent = "Finding top resumes for job...";
      console.log(`Requesting top resumes for Job ID: ${jobId}`);

      try {
          // Updated endpoint to reflect Blueprint prefixes
          const response = await fetch(`/jobs/match-resumes/${jobId}?top_n=5`);
          const data = await response.json();
          console.log("Top resumes response:", data);

          if (response.ok) {
              let resultText = `Top resumes for Job ID: ${jobId}\n\n`;
              if (data.top_resume_matches && data.top_resume_matches.length > 0) {
                  data.top_resume_matches.forEach((match, index) => {
                      resultText += `${index + 1}. Candidate Name: ${
                          match.candidate_name || "N/A"
                      }\n`;
                      resultText += `   Resume ID: ${match.resume_id}\n`;
                      resultText += `   Overall Score: ${(
                          match.overall_score * 100
                      ).toFixed(2)}%\n`;
                      resultText += `   Skill Match Score: ${(
                          match.skill_match_score * 100
                      ).toFixed(2)}%\n`;
                      resultText += `   Matched Skills: ${
                          match.matched_skills.join(", ") || "None"
                      }\n\n`;
                  });
              } else {
                  resultText +=
                      "No top matches found for this job description or no resumes uploaded yet.";
              }
              matchResultContent.textContent = resultText;
              // Update token usage if provided by the backend (for the overall operation if available)
              // Note: Your backend for match-resumes doesn't seem to return token usage directly
              // If it did, you would use data.token_usage or similar.
              updateTokenUsage(
                  matchTokenUsageSection,
                  matchPromptTokens,
                  matchCompletionTokens,
                  matchTotalTokens,
                  null // Assuming no token usage returned for this endpoint
              );
          } else {
              matchResultContent.textContent = `Error: ${
                  data.error || "Unknown error"
              }`;
              matchResultContent.style.color = "red";
              console.error("API Error for top resumes:", data);
          }
      } catch (error) {
          matchResultContent.textContent = `Network error: ${error.message}`;
          matchResultContent.style.color = "red";
          console.error("Fetch error for top resumes:", error);
      } finally {
          toggleLoading(findTopResumesBtn, findTopResumesSpinner, false);
      }
  });

  // Event Listener for Generate Interview Questions Button
  generateQuestionsBtn.addEventListener("click", async () => {
      console.log("Generate Interview Questions button clicked.");
      const resumeId = matchResumeIdInput.value.trim();
      const jobId = matchJobIdInput.value.trim();

      interviewQuestionsResultDiv.classList.remove("hidden"); // Ensure questions result div is visible
      interviewQuestionsContent.style.color = "inherit"; // Reset text color
      updateTokenUsage(
          interviewQuestionsTokenUsageSection,
          interviewQuestionsPromptTokens,
          interviewQuestionsCompletionTokens,
          interviewQuestionsTotalTokens,
          null
      ); // Clear previous token usage

      if (!resumeId || !jobId) {
          interviewQuestionsContent.textContent =
              "Please enter both Resume ID and Job ID to generate questions.";
          interviewQuestionsContent.style.color = "red";
          console.warn("Missing Resume ID or Job ID for question generation.");
          return;
      }

      toggleLoading(generateQuestionsBtn, generateQuestionsSpinner, true);
      interviewQuestionsContent.textContent =
          "Generating interview questions... This may take a moment.";
      console.log(
          `Requesting interview questions for Resume ID: ${resumeId}, Job ID: ${jobId}`
      );

      try {
          // Updated endpoint to reflect Blueprint prefixes
          // The route is defined in resume_routes.py as /<resume_id>/generate-interview-questions/<job_id>
          // With the blueprint prefix, it becomes /resumes/<resume_id>/generate-interview-questions/<job_id>
          const response = await fetch(
              `/resumes/${resumeId}/generate-interview-questions/${jobId}`
          );
          const data = await response.json();
          console.log("Generate questions response:", data);

          if (response.ok) {
              if (data.interview_questions) {
                  // Your API returns 'interview_questions'
                  let questionsText = `Generated Interview Questions for Candidate (ID: ${resumeId}) for Job (ID: ${jobId}):\n\n`;
                  // Check if technical_questions and behavioral_questions exist and are arrays
                  if (
                      data.interview_questions.technical_questions &&
                      Array.isArray(data.interview_questions.technical_questions) &&
                      data.interview_questions.technical_questions.length > 0
                  ) {
                      questionsText += `Technical Questions:\n`;
                      data.interview_questions.technical_questions.forEach((q, i) => {
                          questionsText += `${i + 1}. ${q}\n`;
                      });
                  } else {
                      questionsText += "No technical questions generated.\n";
                  }

                  questionsText += `\nBehavioral Questions:\n`;
                  if (
                      data.interview_questions.behavioral_questions &&
                      Array.isArray(data.interview_questions.behavioral_questions) &&
                      data.interview_questions.behavioral_questions.length > 0
                  ) {
                      data.interview_questions.behavioral_questions.forEach((q, i) => {
                          questionsText += `${i + 1}. ${q}\n`;
                      });
                  } else {
                      questionsText += "No behavioral questions generated.\n";
                  }

                  interviewQuestionsContent.textContent = questionsText;
              } else {
                  interviewQuestionsContent.textContent =
                      "No interview questions found in the response. Check logs for details.";
              }
              // Corrected: Use data.llm_token_usage_generation for interview question generation
              updateTokenUsage(
                  interviewQuestionsTokenUsageSection,
                  interviewQuestionsPromptTokens,
                  interviewQuestionsCompletionTokens,
                  interviewQuestionsTotalTokens,
                  data.llm_token_usage_generation
              );
          } else {
              interviewQuestionsContent.textContent = `Error: ${
                  data.error || "Unknown error"
              }`;
              interviewQuestionsContent.style.color = "red";
              console.error("API Error for interview questions:", data);
          }
      } catch (error) {
          interviewQuestionsContent.textContent = `Network error: ${error.message}`;
          interviewQuestionsContent.style.color = "red";
          console.error("Fetch error for interview questions:", error);
      } finally {
          toggleLoading(generateQuestionsBtn, generateQuestionsSpinner, false);
      }
  });

  // Initial clear to set placeholders correctly on load
  clearResumeResults();
  clearJobResults();
  clearJobTextResults(); // Clear the new job text section on load
});
