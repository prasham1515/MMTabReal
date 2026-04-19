from bs4 import BeautifulSoup


def normalize_table(table):
    """
    Given a BeautifulSoup <table> tag, return a new <table> tag where
    every rowspan/colspan > 1 has been expanded into separate cells,
    and borders are applied to the table and its cells.
    """
    cell_map = {}   # (row_idx, col_idx) -> (inner_html, tag_name)
    rows = table.find_all('tr')
    max_cols = 0

    # 1) Read in all cells, record their spans into cell_map
    for r, row in enumerate(rows):
        c = 0
        for cell in row.find_all(['td', 'th']):
            # skip to next free column
            while (r, c) in cell_map:
                c += 1

            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            inner_html = ''.join(str(x) for x in cell.contents)
            tag_name = cell.name

            # fill every covered slot
            for dr in range(rowspan):
                for dc in range(colspan):
                    rr = r + dr
                    cc = c + dc
                    cell_map[(rr, cc)] = (inner_html, tag_name)
                    max_cols = max(max_cols, cc + 1)

            c += colspan

    # 2) Build a brand‑new <table> tag without any spans, adding borders
    total_rows = max((pos[0] for pos in cell_map), default=-1) + 1
    soup = BeautifulSoup("", 'html.parser')
    new_table = soup.new_tag('table', border="1", style="border-collapse: collapse;")

    for rr in range(total_rows):
        new_tr = soup.new_tag('tr')
        for cc in range(max_cols):
            content, tag = cell_map.get((rr, cc), ("", "td"))
            new_cell = soup.new_tag(tag)
            # apply cell border
            new_cell['style'] = "border: 1px solid black;"
            # parse and insert original content
            new_cell.append(BeautifulSoup(content, 'html.parser'))
            new_tr.append(new_cell)
        new_table.append(new_tr)

    return new_table


def process_html(html):
    """
    Parse the given HTML, replace every <table> with its normalized version,
    and return the resulting HTML string.
    """
    soup = BeautifulSoup(html, 'html.parser')
    for table in soup.find_all('table'):
        new_t = normalize_table(table)
        table.replace_with(new_t)
    return str(soup)


if __name__ == "__main__":
    input_file = r"C:\Users\prash\OneDrive\Desktop\Gemini\Board_of_Regents_of_the_University_of_Michigan\Board_of_Regents_of_the_University_of_Michigan.html"
    output_file = r"C:\Users\prash\OneDrive\Desktop\Gemini\Board_of_Regents_of_the_University_of_Michigan\out.html"

    # Read in the original HTML
    with open(input_file, "r", encoding="utf-8") as f:
        html = f.read()

    # Normalize tables and add borders
    new_html = process_html(html)

    # Write out the expanded HTML
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_html)

    print(f"Written expanded tables with borders to:\n  {output_file}")
