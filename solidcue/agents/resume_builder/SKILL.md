# SKILL.md

# Resume Builder Skill

# [PLANNER GUIDANCE] - Execution Workflow

_Target these milestones when generating the task plan:_

Tasks

1. **Source**: Extract Job Description from user input. Refer to `TOOLS.md`
2. **Source**: Load master resume from `resume_agent/source/master`. Refer to `TOOLS.md`
3. **Content Creation**: Apply Resume Strategy under the [RESUME STRATEGY] section below to generate the content of the resume.
4. **Output**: Refer to `TOOLS.md`
5. **Job Tracker Update**: After resume is uploaded, update the job tracker spreadsheet. This is an artifact_generation task — NOT part of synthesis. Refer to `TOOLS.md` Job Tracker Routing section.

# [RESUME STRATEGY]

_Use these rules during the **Content Creation Stage** task._

## Page Budget

**HARD LIMIT: The final resume MUST fit within 2 pages.** This is non-negotiable.
When content exceeds 2 pages, apply these cuts in order:

1. Shorten bullets in low-relevance roles first.
2. Reduce Key Achievements bullets to 3.
3. Trim Technical Projects to top 2.
4. Compress Professional Summary to 3 lines.
   Never sacrifice high-relevance role detail to save space — cut from the bottom of the priority stack.

**BEFORE WRITING CHECK IF JOB DESCRIPTION IS AVAILABLE!**

## Job Description Tailoring Rules

When a job description is provided:

- NEVER copy & paste the exact source file, resume master directly.

You MUST:

- Extract required and preferred skills.
- Identify repeated terminology and core responsibilities.
- Map requirements to verified experience and projects.
- Reorder bullets/sections by relevance.
- Align wording with the job language without copying text verbatim.

Must avoid:

- Adding tools or skills not present in source data.
- Keyword stuffing.
- Forcing unrelated content to appear relevant.

## Resume Creation Workflow

1. **Requirement Mapping:**: Analyze the Job Description to identify required technical skills, role expectations, and domain-specific terminology while preserving factual chronology and verified experience boundaries.
2. **Evidence Extraction:**:Scan source materials to identify factual anchors (titles, dates, tech stacks) that specifically support the mapped requirements.
3. **Impact Synthesis:**: Apply the PAR (Problem-Action-Result) framework to raw experience, ensuring every bullet point connects a technical task to a business outcome.
4. **Priority Weighting:**: Reorder bullet points and project entries so the most relevant evidence for the target role appears at the top of each section.
5. **Fact-Check Validation:**: Verify that no technologies or metrics were hallucinated and that all dates are chronologically consistent before generating the final Markdown.

## Section Rules (In Order)

1. Candidate Name and Information

2. Professional Summary

- Copy and Paste the exact Professional Summary section from the source `resume_agent/source/master`.

3. Skills and Competencies

- Render as a compact block — one line per category using bold category label prefix, comma-separated values. No sub-bullets.
- Group by category (e.g., **Languages:** Python, Go, TypeScript).
- Merge or drop categories with fewer than 2 items into the nearest related category.
- Reorder categories so the most relevant to the target role appear first.
- Include only technologies supported by source materials.
- Avoid keyword stuffing.
- Remove Programming Languages category if it is not stated in the JD.
- Max 4 bullet points.

4. Key Achievements

- Limit to 3-5 high-impact bullets that represent "career-defining" wins.
- HARD LIMIT: Each bullet must be 1-2 lines max (~20-30 words). If longer, compress.
- Use tight PAR structure: "[Action] [technology/method] → [quantified result]". No narrative preamble.
- Quantify Impact: Mandate the use of metrics (e.g., % reduction in latency, $M in value, % automation coverage).
- Focus on Modernization: Prioritize achievements that show a transition from legacy to modern stacks.
- Include at least one non-technical achievement (leadership, stakeholder management, or business growth).
- Group bullets by category label (e.g., "Infrastructure:", "AI/ML:", "Leadership:"). Use the label as a bold prefix on each bullet.

5. Professional Experience

- HARD LIMIT per bullet: 1 line (~15-25 words). No multi-line bullets.
- Prefer bullet structure: `Action Verb + Technical Task + Technology + Outcome/Impact`.
- Bullet allocation by relevance:
  - Highly relevant roles: max 4-5 bullets.
  - Moderately relevant roles: max 2 bullets.
  - Low-relevance roles: max 1 bullets. Keep transferable, strip domain detail.
- Focus on impact, ownership, and implementation depth.
- Do not omit any role — but ruthlessly compress irrelevant ones.
- When source data groups bullets by category (e.g., "Backend Engineering:", "DevOps:"), preserve those category labels as bold prefixes. Reorder categories by relevance to the target role. Drop categories that add no signal for the target job — redistribute their strongest bullet into the nearest relevant category if worth keeping.

6. Technical Projects

- HARD Limit to **3-4 high-impact bullets** for each project that represent "career-defining" wins.
- Include technologies naturally and only when supported by source data.
- Group project bullets by domain category (e.g., "Architecture:", "Deployment:", "AI/ML:") using bold prefixes. Order categories by relevance to the target role.

7. Education and Certifications

- Keep concise and relevant.

## Standard Resume Template:

URLs must be plain text — NEVER use markdown link syntax like [text](url).
Every placeholder below (wrapped in {{ }}) MUST be replaced with real data from source materials. NEVER output {{ }} placeholders, example URLs, or bracket placeholders in the final resume.
Always start the first line with `#` (H1)

```
# {{full_name}} | {{profile_role}}
{{location}} | {{phone}} | {{email}}
Portfolio: {{portfolio_url}}
LinkedIn: {{linkedin_url}} | GitHub: {{github_url}}

## Professional Summary
- Copy and Paste the exact Professional Summary section from the source.

## Skills and Competencies
- **{{category}}:** {{skill}}, {{skill}}, {{skill}}
- **{{category}}:** {{skill}}, {{skill}}

## Key Achievements
- **{{category}}:** {{achievement}}
- **{{category}}:** {{achievement}}

## Professional Experience
**{{company_name}}** - **{{job_title}}** | {{start_date}} – {{end_date}}
- **{{category}}:** {{bullet}}
- **{{category}}:** {{bullet}}

## Technical Projects
**{{project_name}}** | {{repo_url}}
{{headline}}
- **{{category}}:** {{bullet}}
- **{{category}}:** {{bullet}}

## Education and Certifications
**{{institution}}** | {{course_name}}
**{{institution}}** | {{course_name}}

```

- HARD RULE: Do not add filename to the generated content.

## Validation Checklist

Before final output, verify:

- Dates are consistent.
- Technologies and claims are source-backed.
- No fabricated projects, metrics, or responsibilities.
- Bullets are concise and non-duplicative.
- Resume is role-specific and ATS-readable.
- Formatting follows the selected template requirements.

# [ARTIFACT DELIVERY]

## File System & Naming

Storage Path: Save tailored resumes to `resume_agent/generated_resumes/`

Render Filename in this format:

`YYYYMMDD_{{full_name}}_{{company}}_{{role}}_resume`

- Replace all `{{}}` placeholders with actual values gathered during content creation.
- Example: `20260515_john_smith_accenture_genai_engineer_resume`

- `YYYYMMDD` - Date
- `full_name` - Extract it from the Personal Information section at the top of the resume content. Never omit
  it.
- `company` - Company name extract from JD
- `role` - Role the company is hiring extract from JD
- Lowercase only
- Underscores between words
- No spaces or special characters

# [JOB TRACKER RULES]

## Job Tracker Update Rules

When a resume is generated for an application, update:

- `resume_agent/job_tracker/ai_job_tracker`

Add one row with:

- Date Applied: Date of this task is created
- Application Status: Default as Drafting
- Company: The company names stated in JD
- Role Title: The role stated in JD
- Posting: The JD link provided by user
- Source: The source of the JD (eg: LinkedIn)
- Resume URL: url of the generated resume
- Location: Location of the posted job (eg: Singapore, Hong Kong)
- Work Arrangement: [On Site, Work From Home, Hybrid] get this value from JD. If not listed in JD, default to On Site
- Salary Range: Get information from JD if listed
- Interview Stage: Leave Blank
- Last Updated: Leave Blank
- Next Follow Up Date: Leave Blank
- Recruiter Name: Leave Blank
- Recruiter Contact: Leave Blank
- Notes: Leave Blank
- Outcome: Leave Blank

Never claim tracker updates succeeded unless the tool returns success.
