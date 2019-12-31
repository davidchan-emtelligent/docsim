# docsim
legal document paragraph similarity

## Installation

To use `virtualenv` to manage dependencies, first setup a virtualenv environment:

    python3 -m venv venv
    source venv/bin/activate

Then within your environment install the requirements:

    pip install .

If you'd like to be able to edit the code that has been installed without
having to re-install it, run

    pip install -e .

## To run command line

    doc-sim -i data/pdfs -f pdf -o html.html -k data/terms.txt -m model
    doc-sim -r --source_para_id 10,11

