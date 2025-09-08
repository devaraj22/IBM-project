import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np

# Parse the provided data with improved organization
data = {
    "nodes": [
        {"id": "streamlit", "label": "Streamlit App", "layer": "frontend", "type": "web_app", "symbol": "square"},
        {"id": "fastapi", "label": "FastAPI Gateway", "layer": "api", "type": "api_gateway", "symbol": "diamond"},
        {"id": "drug_service", "label": "Drug Interact", "layer": "service", "type": "business_logic", "symbol": "circle"},
        {"id": "nlp_service", "label": "NLP Process", "layer": "service", "type": "ai_service", "symbol": "circle"},
        {"id": "dosage_service", "label": "Dosage Calc", "layer": "service", "type": "business_logic", "symbol": "circle"},
        {"id": "alternative_service", "label": "Alternative", "layer": "service", "type": "business_logic", "symbol": "circle"},
        {"id": "sqlite_db", "label": "SQLite DBs", "layer": "data", "type": "database", "symbol": "square"},
        {"id": "rxnorm_api", "label": "RxNorm API", "layer": "external", "type": "external_api", "symbol": "triangle-up"},
        {"id": "openfda_api", "label": "OpenFDA API", "layer": "external", "type": "external_api", "symbol": "triangle-up"},
        {"id": "watson_api", "label": "Watson NLU", "layer": "external", "type": "ai_api", "symbol": "star"},
        {"id": "gemini_api", "label": "Gemini AI", "layer": "external", "type": "ai_api", "symbol": "star"},
        {"id": "huggingface", "label": "HuggingFace", "layer": "external", "type": "ml_models", "symbol": "star"},
        {"id": "security", "label": "Security", "layer": "infrastructure", "type": "security", "symbol": "diamond"},
        {"id": "docker", "label": "Docker", "layer": "infrastructure", "type": "containerization", "symbol": "hexagon"}
    ]
}

# Define layer hierarchy, colors, and improved spacing
layer_colors = {
    "frontend": "#1FB8CD",      # Strong cyan
    "api": "#DB4545",           # Bright red  
    "service": "#2E8B57",       # Sea green
    "data": "#5D878F",          # Cyan
    "external": "#D2BA4C",      # Moderate yellow
    "infrastructure": "#B4413C" # Moderate red
}

layer_y_positions = {
    "frontend": 6,
    "api": 5,
    "service": 4,
    "data": 3,
    "external": 2,
    "infrastructure": 1
}

layer_names = {
    "frontend": "Frontend Layer",
    "api": "API Layer", 
    "service": "Service Layer",
    "data": "Data Layer",
    "external": "External APIs",
    "infrastructure": "Infrastructure"
}

# Create improved node positions with better spacing
node_positions = {}
layer_groups = {}

# Group nodes by layer
for node in data["nodes"]:
    layer = node["layer"]
    if layer not in layer_groups:
        layer_groups[layer] = []
    layer_groups[layer].append(node)

# Assign positions with improved spacing
for layer, nodes in layer_groups.items():
    y = layer_y_positions[layer]
    num_nodes = len(nodes)
    
    # Better horizontal spacing
    if num_nodes == 1:
        x_positions = [0]
    elif num_nodes == 2:
        x_positions = [-2, 2]
    elif num_nodes == 3:
        x_positions = [-3, 0, 3]
    elif num_nodes == 4:
        x_positions = [-4.5, -1.5, 1.5, 4.5]
    else:
        x_positions = [(i - (num_nodes - 1) / 2) * 2.5 for i in range(num_nodes)]
    
    for i, node in enumerate(nodes):
        node_positions[node["id"]] = (x_positions[i], y)

# Create layer background rectangles
layer_shapes = []
for layer, y_pos in layer_y_positions.items():
    layer_shapes.append(
        dict(
            type="rect",
            x0=-6, x1=6,
            y0=y_pos-0.4, y1=y_pos+0.4,
            fillcolor=layer_colors[layer],
            opacity=0.1,
            line=dict(width=0)
        )
    )

# Create edge traces with arrows
edge_traces = []
connections = [
    ("streamlit", "fastapi"),
    ("fastapi", "drug_service"),
    ("fastapi", "nlp_service"), 
    ("fastapi", "dosage_service"),
    ("fastapi", "alternative_service"),
    ("drug_service", "sqlite_db"),
    ("nlp_service", "sqlite_db"),
    ("dosage_service", "sqlite_db"),
    ("alternative_service", "sqlite_db"),
    ("drug_service", "rxnorm_api"),
    ("drug_service", "openfda_api"),
    ("nlp_service", "watson_api"),
    ("nlp_service", "gemini_api"),
    ("nlp_service", "huggingface"),
    ("security", "fastapi"),
    ("security", "sqlite_db"),
    ("docker", "streamlit"),
    ("docker", "fastapi")
]

for from_id, to_id in connections:
    from_pos = node_positions[from_id]
    to_pos = node_positions[to_id]
    
    # Create curved arrows
    mid_x = (from_pos[0] + to_pos[0]) / 2
    mid_y = (from_pos[1] + to_pos[1]) / 2
    
    edge_trace = go.Scatter(
        x=[from_pos[0], mid_x, to_pos[0]],
        y=[from_pos[1], mid_y, to_pos[1]],
        mode='lines',
        line=dict(width=2, color='rgba(70,70,70,0.6)'),
        hoverinfo='none',
        showlegend=False
    )
    edge_traces.append(edge_trace)
    
    # Add arrow head
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1] 
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        # Normalize and create arrow
        dx_norm = dx / length * 0.2
        dy_norm = dy / length * 0.2
        
        arrow_trace = go.Scatter(
            x=[to_pos[0] - dx_norm, to_pos[0], to_pos[0] - dx_norm],
            y=[to_pos[1] - dy_norm + 0.1, to_pos[1], to_pos[1] - dy_norm - 0.1],
            mode='lines',
            line=dict(width=2, color='rgba(70,70,70,0.8)'),
            fill='toself',
            fillcolor='rgba(70,70,70,0.8)',
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(arrow_trace)

# Symbol mapping for different node types
symbol_map = {
    "square": "square",
    "circle": "circle", 
    "diamond": "diamond",
    "triangle-up": "triangle-up",
    "star": "star",
    "hexagon": "hexagon"
}

# Create node traces by type for better organization
node_traces = []
for layer, color in layer_colors.items():
    layer_nodes = [node for node in data["nodes"] if node["layer"] == layer]
    
    if not layer_nodes:
        continue
        
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    symbols = []
    
    for node in layer_nodes:
        pos = node_positions[node["id"]]
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_text.append(node["label"])
        symbols.append(symbol_map.get(node.get("symbol", "circle"), "circle"))
        
        # Create detailed hover text
        hover_info = f"<b>{node['label']}</b><br>Type: {node['type']}<br>Layer: {layer.title()}"
        node_hover.append(hover_info)
    
    # Use the most common symbol for the layer
    common_symbol = max(set(symbols), key=symbols.count) if symbols else "circle"
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=50,
            color=color,
            symbol=common_symbol,
            line=dict(width=3, color='white')
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=11, color='white', family="Arial Black"),
        hovertext=node_hover,
        hoverinfo="text",
        name=layer_names[layer],
        showlegend=True
    )
    node_traces.append(node_trace)

# Create figure
fig = go.Figure(data=edge_traces + node_traces)

# Add layer labels on the left
for layer, y_pos in layer_y_positions.items():
    fig.add_annotation(
        x=-7, y=y_pos,
        text=f"<b>{layer_names[layer]}</b>",
        showarrow=False,
        font=dict(size=14, color=layer_colors[layer]),
        xanchor="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=layer_colors[layer],
        borderwidth=1
    )

# Add shapes for layer backgrounds
fig.update_layout(shapes=layer_shapes)

fig.update_layout(
    title="AI Medical Prescription System",
    showlegend=True,
    legend=dict(
        orientation='h', 
        yanchor='bottom', 
        y=1.02, 
        xanchor='center', 
        x=0.5,
        font=dict(size=10)
    ),
    xaxis=dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False,
        range=[-8, 8]
    ),
    yaxis=dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False,
        range=[0.5, 6.5]
    ),
    plot_bgcolor='white',
    annotations=[
        dict(
            text="System architecture with hierarchical layers and data flow connections",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            xanchor='center', yanchor='top',
            font=dict(size=11, color='gray')
        )
    ]
)

fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("ai_medical_system_architecture.png", width=1400, height=900)