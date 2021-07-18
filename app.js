
let viz; 

const url = 

"https://public.tableau.com/views/MarylandCoviddemo/DEMO-MarylandCovidcasesdashboard?"

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

