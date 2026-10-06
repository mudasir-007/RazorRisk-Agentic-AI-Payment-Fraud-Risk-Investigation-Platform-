let networkInstance = null;

function renderGraphTopology(userId) {
    const container = document.getElementById('vis-network-container');
    const noteEl = document.getElementById('graph-truncated-note');
    if (!container) return;

    if (noteEl) noteEl.style.display = 'none';
    container.innerHTML = '<div class="graph-loading">Loading network…</div>';

    fetch(`${API_BASE}/api/v1/graph/topology/${userId}`)
        .then(res => res.json())
        .then(data => {
            if (!data.nodes || data.nodes.length === 0) {
                container.innerHTML = '<div class="graph-empty">No graph network topology found for this user yet — score a transaction for them first.</div>';
                return;
            }

            const nodes = new vis.DataSet(data.nodes);
            const edges = new vis.DataSet(data.edges);

            const visData = { nodes: nodes, edges: edges };

            if (noteEl) noteEl.style.display = data.truncated ? 'block' : 'none';

            // Tuned specifically to stop label/edge overlap on dense fraud-ring
            // clusters (7+ nodes sharing one device/IP used to collapse into an
            // unreadable knot at default vis.js spacing):
            //  - larger springLength + stronger repulsion pushes ring members
            //    into a legible circle around the shared entity instead of a pile
            //  - avoidOverlap reserves space per node so labels don't stack
            //  - edge labels were removed at the data layer (routes_graph.py);
            //    only node labels render, positioned below each node (vadjust)
            const options = {
                nodes: {
                    borderWidth: 1,
                    shadow: { enabled: true, size: 6, x: 0, y: 2, color: 'rgba(0,0,0,0.35)' },
                    font: { color: '#E2E8F0', face: 'Inter', size: 13, strokeWidth: 0 }
                },
                edges: {
                    width: 1.5,
                    shadow: false,
                    smooth: { type: 'continuous', roundness: 0.35 },
                    color: { color: '#37415199', highlight: '#64748B', hover: '#64748B' }
                },
                physics: {
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {
                        gravitationalConstant: -80,
                        centralGravity: 0.008,
                        springLength: 160,
                        springConstant: 0.12,
                        avoidOverlap: 1
                    },
                    stabilization: { iterations: 150 }
                },
                layout: { improvedLayout: true },
                interaction: { hover: true, tooltipDelay: 120, zoomView: true, dragView: true }
            };

            if (networkInstance) {
                networkInstance.destroy();
            }

            networkInstance = new vis.Network(container, visData, options);
            networkInstance.once('stabilizationIterationsDone', () => {
                networkInstance.setOptions({ physics: false });
            });
        })
        .catch(err => {
            console.error("Error rendering graph:", err);
            container.innerHTML = '<div class="graph-empty">Could not load the network graph. Check the Audit Logs tab for details.</div>';
        });
}
