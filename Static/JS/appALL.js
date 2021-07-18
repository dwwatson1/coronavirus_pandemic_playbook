
let viz; 

const url = 

"https://public.tableau.com/views/ALLSTATESDATAMARCHtoDEC2020/Dashboard4?:language=en-US&:display_count=n&:origin=viz_share_link"

const vizContainer = document.getElementById("vizContainer");

const exportPDFButton = document.getElementById("exportPDF");

function initViz() {

    viz = new tableau.Viz(vizContainer, url);

}

function exportToPDF() {

    viz.showExportPDFDialog();
}

exportPDFButton.addEventListener("click", exportToPDF);

document.addEventListener("DOMContentLoaded", initViz);

