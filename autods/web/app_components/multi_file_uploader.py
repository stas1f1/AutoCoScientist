import streamlit as st


def render_multi_uploader(
    label: str,
    key: str,
    types=None,
    help_text: str | None = None,
):
    files = st.file_uploader(
        label,
        key=key,
        type=types,
        accept_multiple_files=True,
    )
    if help_text:
        st.caption(help_text)

    if files:
        st.success(f"Загружено файлов: {len(files)}")
        for file in files:
            size_kb = len(file.getvalue()) / 1024
            st.write(f"• {file.name} ({size_kb:.1f} KB)")

    return files
