import os
import re

from itemadapter import ItemAdapter


class IrishStatutesPipeline:
    def process_item(self, item, spider):
        return item


class HtmlFilePipeline:
    """Persist raw HTML for each scraped act to raw_html/{year}/act_{number}.html."""

    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "irish_statutes", "raw_html")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        url = adapter.get("url", "")
        html = adapter.get("full_text", "")

        if not url or not html:
            return item

        # /eli/2000/act/42/enacted/en/print.html  →  year=2000, number=42
        match = re.search(r"/eli/(\d+)/act/(\d+)/", url)
        if not match:
            return item

        year, number = match.group(1), match.group(2)
        dest_dir = os.path.join(self.BASE_DIR, year)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"act_{number}.html")

        if not os.path.exists(dest_path):
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(html)

        return item
