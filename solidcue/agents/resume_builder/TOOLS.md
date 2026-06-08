```markdown
# TOOLS.md

# Resume Builder Tool Routing

Use this file to decide which tool to call for each phase of the resume workflow.

## Enabled Tools For This Agent

- `drive_list_by_path`
- `drive_download_file`
- `drive_upload_file`
- `drive_ensure_folder_path`
- `create_formatted_word_document_base64`
- `browser_navigate`
- `browser_get_html`
- `sheets_read_values`
- `sheets_get_spreadsheet`
- `sheets_append_values`

Only call tools listed above unless the agent configuration is updated.

## Retrieval Routing

Goal: find and read resume source data from Google Drive.

1. Use `drive_list_by_path` first to locate source files under:

- `resume_agent/source/master`

2. After a source file is identified, use `drive_download_file` to retrieve content.

- For Google Docs (mimeType `application/vnd.google-apps.document`), set `export_mime_type` to `text/plain` so the content is readable text.
- Never export as `application/vnd.openxmlformats-officedocument.wordprocessingml.document` when the purpose is reading content because that returns a binary blob.

3. Use `browser_navigate` before `browser_get_html` when the source is a web page or job posting URL that must be read for tailoring.

Do not skip path listing when file IDs are unknown.

## Artifact Generation Routing

Goal: build a resume document after source retrieval. Follow these exact sequences.

1. Use `create_formatted_word_document_base64` to create the resume file payload.
2. Use `drive_ensure_folder_path` to resolve/create the target Google Drive folder path and get its folder `id`.
3. Use `drive_upload_file` with the generated payload and `parent_id` set to that folder `id`.
4. Only say the resume is finished after the upload is successful.

If creating, folder resolution, or uploading fails, stop and tell the user exactly what failed.

## Job Description Routing

Determine how the job description was provided:

### Option A — User pasted JD text directly (no URL in user input)

The job description content is already present in the user's message.
Do NOT create any source_gathering task for JD extraction. There is no tool to call — the text is already available.
Do NOT call `browser_navigate` or `browser_get_html` — there is no URL to fetch.
The first task in the plan should be the master resume retrieval step (`drive_list_by_path`).
The pasted JD text in the user input is the authoritative job description for tailoring.

### Option B — User provided a job URL

1. Use `browser_navigate` to open the job URL.
2. Use `browser_get_html` to extract job text from the open browser page.
3. Tailor summary, skills emphasis, experience bullets, and project ordering using verified source facts.

If the job URL cannot be read, ask for pasted job text.

## Job Tracker Routing

Goal: update the job tracker spreadsheet after a resume is generated, or create one if it does not exist.

### Locating the Tracker

1. Use `drive_list_by_path` on `resume_agent/job_tracker` to find the tracker spreadsheet.
2. If found, extract the file ID from the listing result.
3. If the folder or spreadsheet does not exist, use `sheets_create_job_tracker` to create one (requires adding this tool to the agent config first).

### Reading Existing Data

1. Use `sheets_read_values` with the tracker spreadsheet ID and `range_name: "Applications!A:A"` to read column A values.

- This returns only cell values as a 2D array with no formatting metadata.
- Do NOT use `sheets_get_spreadsheet` with `include_grid_data: true` for reading cell values because it returns large formatting metadata that can exceed LLM context limits.

2. Count the non-empty rows in the response to determine whether there is an existing table/header and to sanity check append placement.

### Writing New Tracker Rows

1. Use `sheets_append_values` to add each new application row.
2. Use a stable anchor range such as `Applications!A:Z`.
3. Keep `insert_data_option` as `INSERT_ROWS` (default) to avoid overwriting existing rows.
4. Set `value_input_option` to:

- `RAW` for literal text
- `USER_ENTERED` when formulas/date parsing are desired

5. Optionally verify the write with `sheets_read_values` on the relevant range after append.

## Tool Usage Constraints

- Use `drive_list_by_path` before `drive_download_file` unless a file ID is already known.
- Use `browser_navigate` before `browser_get_html` unless the browser page is already open on the target URL.
- Use `browser_get_html` only for web content extraction, not for Drive file retrieval.
- Prefer `sheets_read_values` for value retrieval over `sheets_get_spreadsheet` with grid data.
- Use `sheets_append_values` for inserting new tracker entries.
- Do not claim a tool action succeeded unless the tool returns success.
```
