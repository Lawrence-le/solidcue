from solidcue.tools.stages import infer_tool_stage


def test_infer_tool_stage_classifies_read_tools_as_context() -> None:
    assert infer_tool_stage("drive_search_files") == "context"
    assert infer_tool_stage("drive_download_file") == "context"
    assert infer_tool_stage("scrape_webpage") == "context"


def test_infer_tool_stage_classifies_write_tools_as_artifact() -> None:
    assert infer_tool_stage("docs_create_document") == "artifact"
    assert infer_tool_stage("drive_upload_file") == "artifact"
    assert infer_tool_stage("sheets_write_values") == "artifact"
