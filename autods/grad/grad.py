import asyncio
from pathlib import Path
from typing import Any

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.modules.engine.operations.setup import setup

from autods.constants import REPO_STORAGE
from autods.grad.repository import clone_repository, get_repository_id
from autods.grad.xmlapi import extract_entities
from autods.prompting.prompt_store import prompt_store
from autods.repository_processor.processor import process_repository


class grad:
    @staticmethod
    async def add(url: str):
        await setup()
        xml_api_path = await create_xml_api_doc(url)
        await cognee_add_xml_api(xml_api_path, get_repository_id(url))

    @staticmethod
    async def visualize(path: str = "./grad.html"):
        await setup()
        await visualize_graph(path)
        return path

    @staticmethod
    async def get_dataset(dataset_name: str, default: Any = None):
        await setup()
        user_datasets = await grad.list_datasets()
        for dataset in user_datasets:
            if dataset.name.lower() == dataset_name.lower():
                return dataset
        return default

    @staticmethod
    async def delete_dataset(dataset_name: str):
        await setup()
        dataset = await grad.get_dataset(dataset_name)
        if dataset:
            await cognee.datasets.delete_dataset(dataset.id)

    @staticmethod
    async def list_datasets():
        await setup()
        return await cognee.datasets.list_datasets()

    @staticmethod
    async def ask(url: str, query: str, download: bool = True):
        await setup()
        repo_id = get_repository_id(url)
        dataset = await grad.get_dataset(repo_id)
        if not dataset:
            if download:
                await grad.add(url)
            else:
                return "The library is not yet indexed."
        system_prompt = prompt_store.load("grad.md")
        result = await cognee.search(
            query_text=query,
            dataset_ids=[dataset.id],
            query_type=cognee.SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
            system_prompt=system_prompt,
        )
        if isinstance(result, list):
            if len(result) == 0:
                return "No results found."
            else:
                return "\n".join(
                    [
                        str(item.get("search_result", ["No results found."])[0])
                        for item in result
                    ]
                )
        else:
            return result.result if result else "No results found."


async def create_xml_api_doc(url: str) -> Path:
    repo_path = Path(REPO_STORAGE) / get_repository_id(url)
    xml_api_filename = "api.xml"
    xml_api_path = repo_path / xml_api_filename

    if not xml_api_path.exists():
        # Create new XML API
        if not repo_path.exists():
            clone_repository(url, repo_path)
        await process_repository(
            repository_path=str(repo_path),
            output_file=xml_api_filename,
        )
    return xml_api_path


def split_xml_api(xml_api_path: Path):
    classes, methods, functions, examples = extract_entities(xml_api_path)
    return [*classes, *methods, *functions, *examples]


async def cognee_add_xml_api(xml_api_path: Path, dataset_name: str):
    custom_prompt = """
    Extract methods, functions and classes as entities, add their parameters to description.
    Connect classes to methods with the relationship "has_method".
    Connect methods to classes with the relationship "belongs_to".
    Connect examples to methods, classes, and functions that are used in this example with the relationship "is_used".
    """
    documents = split_xml_api(xml_api_path)
    await cognee.add(
        documents,
        dataset_name=dataset_name,
        preferred_loaders=["text_loader"],
        data_per_batch=1,
    )
    await cognee.cognify(data_per_batch=1, custom_prompt=custom_prompt)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Graph RAG API Doc")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_repo_parser = subparsers.add_parser(
        "add", help="Add repository to knowledge graph"
    )
    add_repo_parser.add_argument("url", help="Repository URL")

    delete_repo_parser = subparsers.add_parser(
        "delete", help="Delete repository to knowledge graph"
    )
    delete_repo_parser.add_argument("url", help="Repository URL")

    ask_repo_parser = subparsers.add_parser(
        "ask", help="Query repository knowledge graph"
    )
    ask_repo_parser.add_argument("url", help="Repository URL")
    ask_repo_parser.add_argument("query", help="Search query")

    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize knowledge graph"
    )

    list_repo_parser = subparsers.add_parser("list", help="List saved repositories")

    args = parser.parse_args()

    if args.command == "add":
        try:
            asyncio.run(grad.add(args.url))
            print(f"Successfully added repository: {args.url}")
        except Exception as e:
            print(f"Error adding repository: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "delete":
        try:
            asyncio.run(grad.delete_dataset(get_repository_id(args.url)))
            print(f"Successfully deleted repository: {args.url}")
        except Exception as e:
            print(f"Error deleting repository: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "ask":
        try:
            result = asyncio.run(grad.ask(args.url, args.query))
            print(result)
        except Exception as e:
            print(f"Error querying repository: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "list":
        try:
            result = asyncio.run(grad.list_datasets())
            print(result)
        except Exception as e:
            print(f"Error querying repository: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "visualize":
        try:
            path = asyncio.run(grad.visualize())
            print(f"Successfully visualized knowledge graph to {path}")
        except Exception as e:
            print(f"Error visualizing knowledge graph: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
