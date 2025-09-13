# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Create Monad that take a Deployer and make it TilerAware
# Define Tiler Obj centralize all tilling related functionalities for a given deployer.
# Like Template-T-Obj mapping, propagate cst, graph edition, etc


def getSubplotHtml(figJson: str, memoryLevelName: str) -> str:
    return f"""
        <div class="resizable-container">
            <div class="plot" id="plot-{memoryLevelName}"></div>
        </div>
        <script>
            (function() {{
                var fig = {figJson};
                Plotly.newPlot("plot-{memoryLevelName}", fig.data, fig.layout);

                var container = document.getElementById("plot-{memoryLevelName}").parentNode;
                const resizeObserver = new ResizeObserver(() => {{
                    Plotly.relayout("plot-{memoryLevelName}", {{
                        width: container.clientWidth,
                        height: container.clientHeight
                    }});
                }});
                resizeObserver.observe(container);
            }})();
        </script>
    """


def getHtmlMemoryAllocationVisualisation(subplot_html_template: str) -> str:
    return f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Deeploy Memory Allocation - Provided By PULP Platform</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    .resizable-container {{
                        width: 90%;
                        height: 400px;
                        resize: both;
                        overflow: hidden;
                        border: 1px light grey;
                        margin-bottom: 0px;
                        padding: 0px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }}

                    .plot {{
                        width: 100%;
                        height: 100%;
                    }}
                </style>
            </head>
            <body>
                <h1 style="text-align: center;">Deeploy Memory Allocation - Provided By PULP Platform</h1>
                {subplot_html_template}
            </body>
            </html>
    """
