# Forestry Management

The pipeline for surveying a plot of land can involve substantial training, hiring time, and expenses for manual or FPV flights with a drone. This can also introduce the risk of estimates that miss key data points or damaged equipment under unknown environmental conditions. Repeating this work across new boundaries requires further planning and pilot time, while a cell-by-cell record of what was checked would make the survey easier to assess and repeat.

Construction and land development need to know which trees sit inside of a pad or right-of-way before cutting, and where potential wet areas exist for further environmental assessment. Owners who want to reshape a substantial amount of terrain also need the same inventory. Forestry in this case provides the location, type and counts of the trees. In applications such as flood response, where water is visible and which cells of the plot were never seen are also useful information.

The point of this work is to automate that survey pass so that manual effort and repeat-survey time can fall. Given a user-drawn boundary, the aircraft would autonomously produce an on-device report in three layers: trees, water and potential wet areas and wildlife, without requiring a human to review a continuous video stream.

Autonomy here means that the vehicle spends flight time on high-value cells: unseen areas, cells that have not been adequately checked, and detections that need another view. Confirmed geotagged objects can be treated as recorded, while the surrounding cell is assessed separately for coverage. The report accounts for each cell regardless of density.

> This repo is still a work in progress, and as such code may not be final.