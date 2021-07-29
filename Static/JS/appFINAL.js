
let viz; 

const url = 

"https://public.tableau.com/views/THECOVIDPLAYBOOKDASHBOARD/THECOVIDPLAYBOOKDASHBOARD?:language=en-US&:display_count=n&:origin=viz_share_link"

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

