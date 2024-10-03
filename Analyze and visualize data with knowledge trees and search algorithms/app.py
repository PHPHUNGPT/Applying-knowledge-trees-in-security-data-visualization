import streamlit as st
import pandas as pd
import os
import io
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import heapq
import numpy as np
import re
from networkx.algorithms.community import greedy_modularity_communities, girvan_newman

# Đường dẫn thư mục chứa các tệp CSV cho hai thiết bị
device1_directory = "Thiết bị 1"
device2_directory = "Thiết bị 2"

# Hàm đọc dữ liệu CSV một cách mạnh mẽ
def read_csv_with_fallbacks(path):
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        st.warning(f"Lỗi phân tích cú pháp khi đọc tệp {path}. Thử đọc lại bằng cách bỏ qua các dòng lỗi.")
        try:
            return pd.read_csv(path, on_bad_lines='skip')
        except pd.errors.ParserError:
            st.warning(f"Lỗi tiếp tục xảy ra khi đọc tệp {path}. Thử đọc thủ công và xử lý lỗi.")
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            content = content.replace('\r\n', '\n')
            lines = content.split('\n')
            lines = [line for line in lines if len(line.split(',')) == len(lines[0].split(','))]
            return pd.read_csv(io.StringIO('\n'.join(lines)))

# Lấy danh sách các tệp CSV trong các thư mục thiết bị
device1_files = {
    os.path.splitext(filename)[0]: os.path.join(device1_directory, filename)
    for filename in os.listdir(device1_directory)
    if filename.endswith(".csv")
}

device2_files = {
    os.path.splitext(filename)[0]: os.path.join(device2_directory, filename)
    for filename in os.listdir(device2_directory)
    if filename.endswith(".csv")
}

# Đọc và lưu các DataFrame cho mỗi thiết bị
device1_dataframes = {}
device2_dataframes = {}

for name, path in device1_files.items():
    try:
        device1_dataframes[name] = read_csv_with_fallbacks(path)
    except Exception as e:
        st.warning(f"Không thể đọc tệp {name}: {e}")

for name, path in device2_files.items():
    try:
        device2_dataframes[name] = read_csv_with_fallbacks(path)
    except Exception as e:
        st.warning(f"Không thể đọc tệp {name}: {e}")

# Gộp dữ liệu từ hai thiết bị
def merge_data(df1, df2):
    merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    return merged_df

merged_dataframes = {}

for name1, df1 in device1_dataframes.items():
    for name2, df2 in device2_dataframes.items():
        merged_df = merge_data(df1, df2)
        merged_dataframes[f"{name1}_merged_with_{name2}"] = merged_df

# Lưu các DataFrame đã hợp nhất thành các tệp CSV mới
output_directory = "output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for name, df in merged_dataframes.items():
    output_path = os.path.join(output_directory, f"{name}.csv")
    df.to_csv(output_path, index=False)

st.write("Đã gộp dữ liệu và lưu các tệp CSV mới vào thư mục:", output_directory)

# Hàm xây dựng cây tri thức từ các biến do người dùng chọn
def build_knowledge_tree(df, selected_columns):
    G = nx.DiGraph()
    added_edges = set()

    for index, row in df.iterrows():
        for i in range(len(selected_columns) - 1):
            source = f"{selected_columns[i]}: {row[selected_columns[i]]}"
            target = f"{selected_columns[i+1]}: {row[selected_columns[i+1]]}"
            if pd.notna(row[selected_columns[i]]) and pd.notna(row[selected_columns[i+1]]):
                edge = (source, target)
                if edge not in added_edges:
                    G.add_edge(source, target)
                    added_edges.add(edge)
    
    return G

# Hàm tìm và lưu các điểm chung giữa hai bộ dữ liệu
def find_common_points(df1, df2, selected_columns):
    common_points = []
    displayed_points = set()

    for col in selected_columns:
        common_values = pd.Series(list(set(df1[col].dropna()) & set(df2[col].dropna())))
        for value in common_values:
            if value not in displayed_points:
                common_points.append(f"{col}: {value}")
                displayed_points.add(value)
    
    return common_points

def save_and_plot_common_points(common_points):
    # Lưu vào CSV
    common_points_df = pd.DataFrame(common_points, columns=["Common Points"])
    common_points_df.to_csv("common_points.csv", index=False)
    
    # Trực quan hóa bằng biểu đồ
    fig = px.bar(common_points_df, y="Common Points", title="Common Points")
    st.plotly_chart(fig)

# Hàm vẽ đồ thị tương tác 3D hoặc 2D với Plotly và trả về danh sách các nút
def plot_interactive_tree(tree, pos=None, layout='spring', display_mode='3D', path=None, node_colors=None, node_sizes=None):
    if len(tree) == 0:
        st.warning("Cây tri thức trống hoặc không có dữ liệu để vẽ.")
        return None, []

    is_3d = (display_mode == '3D')

    if pos is None:
        if layout == 'spring':
            pos = nx.spring_layout(tree, dim=3 if is_3d else 2)
        elif layout == 'circular':
            pos = nx.circular_layout(tree, dim=3 if is_3d else 2)
        elif layout == 'shell':
            if is_3d:
                st.warning("Shell layout chỉ hỗ trợ 2D, chuyển sang hiển thị dưới dạng 2D.")
            pos = nx.shell_layout(tree)  # Shell layout chỉ hỗ trợ 2D
            is_3d = False
        elif layout == 'spectral':
            pos = nx.spectral_layout(tree, dim=3 if is_3d else 2)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(tree, dim=3 if is_3d else 2)

    # Bắt đầu thay đổi thang màu tại đây
    colorscale = 'YlOrRd'  # Bạn có thể thay đổi thang màu tại đây, ví dụ: 'Bluered', 'YlOrRd', 'Viridis', v.v.
    
    if is_3d:
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in tree.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color_values = []
        node_size_values = []

        for node in tree.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node)
            node_color_values.append(node_colors[node] if node_colors else 0)
            node_size_values.append(node_sizes[node] * 10 if node_sizes else 8)

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=node_text,
            hoverinfo='text',
            textposition='top center',
            marker=dict(
                showscale=True,
                colorscale=colorscale,  # Điều chỉnh thang màu ở đây
                size=node_size_values,
                color=node_color_values,
                colorbar=dict(
                    thickness=15,
                    title='Node Community',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            path_x = []
            path_y = []
            path_z = []
            for edge in path_edges:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])
                path_z.extend([z0, z1, None])

            path_trace = go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                line=dict(width=4, color='red'),
                hoverinfo='none',
                mode='lines')

            fig = go.Figure(data=[edge_trace, node_trace, path_trace])
        else:
            fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title='Cây tri thức từ các biến đã chọn (3D)',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            )
        )
        st.plotly_chart(fig)
    else:
        edge_x = []
        edge_y = []
        for edge in tree.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_color_values = []
        node_size_values = []

        for node in tree.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color_values.append(node_colors[node] if node_colors else 0)
            node_size_values.append(node_sizes[node] * 10 if node_sizes else 8)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            hoverinfo='text',
            textposition='top center',
            marker=dict(
                showscale=True,
                colorscale=colorscale,  # Điều chỉnh thang màu ở đây
                size=node_size_values,
                color=node_color_values,
                colorbar=dict(
                    thickness=15,
                    title='Node Community',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            path_x = []
            path_y = []
            for edge in path_edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])

            path_trace = go.Scatter(
                x=path_x, y=path_y,
                line=dict(width=4, color='red'),
                hoverinfo='none',
                mode='lines')

            fig = go.Figure(data=[edge_trace, node_trace, path_trace])
        else:
            fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title='Cây tri thức từ các biến đã chọn (2D)',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40)
        )
        st.plotly_chart(fig)

    # Hiển thị danh sách các nút
    st.subheader("Danh sách các nút trong biểu đồ:")
    st.write(node_text)

    return pos, node_text

# Hàm lưu các cặp nút có thể đạt được vào CSV và trực quan hóa
def save_and_plot_reachable_pairs(reachable_pairs):
    # Lưu vào CSV
    reachable_pairs_df = pd.DataFrame(reachable_pairs, columns=["Start Node", "Goal Node"])
    reachable_pairs_df.to_csv("reachable_pairs.csv", index=False)
    
    # Trực quan hóa bằng biểu đồ
    fig = px.bar(reachable_pairs_df, x="Start Node", y="Goal Node", title="Reachable Pairs")
    st.plotly_chart(fig)

# Hàm tính chi phí heuristic là hàm Euclid
def euclidean_distance(pos, start, goal):
    start_pos = pos[start]
    goal_pos = pos[goal]

    # Kiểm tra nếu chỉ có 2 giá trị (2D)
    if len(start_pos) == 2:
        x1, y1 = start_pos
        x2, y2 = goal_pos
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    else:
        x1, y1, z1 = start_pos
        x2, y2, z2 = goal_pos
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

# Thuật toán A* với heuristic là hàm Euclid
def a_star_search(tree, pos, start, goal):
    if start not in tree or goal not in tree:
        return None

    queue = [(0, start)]
    costs = {start: 0}
    paths = {start: [start]}
    
    while queue:
        (cost, node) = heapq.heappop(queue)
        
        if node == goal:
            return paths[node]
        
        for neighbor in tree.neighbors(node):
            new_cost = costs[node] + 1  # Mỗi bước đi có chi phí là 1
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                priority = new_cost + euclidean_distance(pos, neighbor, goal)
                heapq.heappush(queue, (priority, neighbor))
                paths[neighbor] = paths[node] + [neighbor]
    
    return None

# Thuật toán Greedy Best-First Search
def gbfs_search(tree, pos, start, goal):
    if start not in tree or goal not in tree:
        return None

    queue = [(euclidean_distance(pos, start, goal), start)]
    paths = {start: [start]}
    visited = set()

    while queue:
        (_, node) = heapq.heappop(queue)
        
        if node == goal:
            return paths[node]

        visited.add(node)
        
        for neighbor in tree.neighbors(node):
            if neighbor not in visited:
                heapq.heappush(queue, (euclidean_distance(pos, neighbor, goal), neighbor))
                paths[neighbor] = paths[node] + [neighbor]
    
    return None

# Thuật toán Beam Search
def beam_search(tree, pos, start, goal, beam_width):
    if start not in tree or goal not in tree:
        return None

    queue = [(euclidean_distance(pos, start, goal), start)]
    paths = {start: [start]}

    while queue:
        next_queue = []
        for (_, node) in queue:
            if node == goal:
                return paths[node]
            for neighbor in tree.neighbors(node):
                if neighbor not in paths:
                    new_path = paths[node] + [neighbor]
                    next_queue.append((euclidean_distance(pos, neighbor, goal), neighbor))
                    paths[neighbor] = new_path
        
        # Giữ lại beam_width nút tốt nhất
        next_queue.sort()
        queue = next_queue[:beam_width]
    
    return None

# Hàm tìm kiếm tự động tất cả các cặp nút và lọc các nút có thể kết nối với nhau
def automatic_search(tree, pos, algorithm='A*'):
    reachable_pairs = []

    nodes = list(tree.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            start_node = nodes[i]
            goal_node = nodes[j]
            if start_node == goal_node:
                continue  # Bỏ qua nếu nút bắt đầu và nút đích giống nhau

            if algorithm == 'A*':
                path = a_star_search(tree, pos, start_node, goal_node)
            elif algorithm == 'Greedy Best-First Search':
                path = gbfs_search(tree, pos, start_node, goal_node)
            elif algorithm == 'Beam Search':
                beam_width = 2  # Giá trị mặc định hoặc có thể lấy từ người dùng
                path = beam_search(tree, pos, start_node, goal_node, beam_width)
            if path:
                reachable_pairs.append((start_node, goal_node))
    
    return reachable_pairs

# Hàm lọc các nhánh liên quan đến đường đi
def filter_subgraph_around_path(tree, path, distance=2):
    sub_nodes = set(path)
    for node in path:
        neighbors = nx.single_source_shortest_path_length(tree, node, cutoff=distance)
        sub_nodes.update(neighbors.keys())
    return tree.subgraph(sub_nodes)

# Hàm lưu đường đi tìm được vào CSV
def save_paths_to_csv(paths, filename):
    df = pd.DataFrame(paths, columns=['Start Node', 'Goal Node', 'Path'])
    df.to_csv(filename, index=False)
    st.write(f'Đã lưu các đường đi vào file {filename}')

# Hàm lưu điểm chung vào CSV
def save_common_points_to_csv(common_points, filename):
    df = pd.DataFrame(common_points, columns=['Common Points'])
    df.to_csv(filename, index=False)
    st.write(f'Đã lưu các điểm chung vào file {filename}')

# Hàm phân tích tập trung
def centrality_analysis(tree):
    degree_centrality = nx.degree_centrality(tree)
    betweenness_centrality = nx.betweenness_centrality(tree)
    closeness_centrality = nx.closeness_centrality(tree)

    centrality_df = pd.DataFrame({
        'Node': list(degree_centrality.keys()),
        'Degree Centrality': list(degree_centrality.values()),
        'Betweenness Centrality': list(betweenness_centrality.values()),
        'Closeness Centrality': list(closeness_centrality.values())
    })

    st.subheader('Phân tích tập trung')
    st.dataframe(centrality_df)

    # Trực quan hóa phân tích tập trung bằng biểu đồ cột
    fig = px.bar(centrality_df, x='Node', y=['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality'],
                 title='Centrality Analysis')
    st.plotly_chart(fig)

    # Biểu đồ phân phối Degree
    fig = px.histogram(centrality_df, x='Degree Centrality', nbins=20,
                       title='Biểu Đồ Phân Phối Degree Centrality')
    st.plotly_chart(fig)

    return centrality_df

# Hàm phát hiện cộng đồng với phương pháp Louvain
def community_detection(tree):
    louvain_communities = list(greedy_modularity_communities(tree))
    community_map = {node: idx for idx, community in enumerate(louvain_communities) for node in community}

    st.subheader('Phát hiện cộng đồng (Louvain Method)')
    community_df = pd.DataFrame({
        'Node': list(community_map.keys()),
        'Community': list(community_map.values())
    })
    st.dataframe(community_df)

    return community_df

# Hàm phát hiện cộng đồng với phương pháp Girvan-Newman
def community_detection_girvan_newman(tree):
    comp = girvan_newman(tree)
    limited = [sorted(c) for c in next(comp)]
    community_map = {node: idx for idx, community in enumerate(limited) for node in community}

    st.subheader('Phát hiện cộng đồng (Girvan-Newman Method)')
    community_df = pd.DataFrame({
        'Node': list(community_map.keys()),
        'Community': list(community_map.values())
    })
    st.dataframe(community_df)

    return community_df

# Hàm tính toán các phép đo mạng
def network_metrics(tree):
    undirected_tree = tree.to_undirected()
    density = nx.density(undirected_tree)
    avg_path_length = nx.average_shortest_path_length(undirected_tree) if nx.is_connected(undirected_tree) else 'N/A'
    clustering_coeff = nx.average_clustering(undirected_tree)

    metrics_df = pd.DataFrame({
        'Metric': ['Density', 'Average Path Length', 'Clustering Coefficient'],
        'Value': [density, avg_path_length, clustering_coeff]
    })

    st.subheader('Phép đo mạng')
    st.dataframe(metrics_df)

    return metrics_df

# Xây dựng ứng dụng Streamlit
st.title('Ứng dụng Phân Tích và Trực Quan Hóa Dữ Liệu An Ninh')

# Hiển thị dữ liệu
st.subheader('Dữ liệu gốc')

# Lựa chọn bộ dữ liệu
selected_dataset1 = st.selectbox("Chọn bộ dữ liệu cho Thiết bị 1", list(device1_dataframes.keys()))
selected_dataset2 = st.selectbox("Chọn bộ dữ liệu cho Thiết bị 2", list(device2_dataframes.keys()))

# Lựa chọn các biến (cột)
df_selected1 = device1_dataframes[selected_dataset1]
df_selected2 = device2_dataframes[selected_dataset2]
selected_columns = st.multiselect("Chọn các biến để vẽ cây tri thức", df_selected1.columns.tolist())

if selected_columns:
    # Gộp dữ liệu
    merged_df = merge_data(df_selected1, df_selected2)

    # Xây dựng cây tri thức
    st.subheader(f'Cây tri thức từ các biến đã chọn cho bộ dữ liệu {selected_dataset1} và {selected_dataset2}')
    knowledge_tree = build_knowledge_tree(merged_df, selected_columns)

    # Chọn hiển thị 2D hoặc 3D
    display_mode = st.radio("Chọn chế độ hiển thị", ('3D', '2D'))

    # Lựa chọn layout
    layout = st.selectbox("Chọn layout cho cây tri thức", ["spring", "circular", "shell", "spectral", "kamada_kawai"], index=4)
    
    # Vẽ cây tri thức và nhận vị trí các nút
    pos, all_nodes = plot_interactive_tree(knowledge_tree, None, layout, display_mode)

    if all_nodes:
        # Tìm kiếm tự động để lọc các cặp nút có thể kết nối với nhau
        st.subheader('Tìm kiếm tự động các nút có đường đi')
        selected_algorithm = st.selectbox('Chọn thuật toán tìm kiếm', ['A*', 'Greedy Best-First Search', 'Beam Search'])
        reachable_pairs = automatic_search(knowledge_tree, pos, selected_algorithm)

        if reachable_pairs:
            save_and_plot_reachable_pairs(reachable_pairs)
        else:
            st.write('Không tìm thấy các cặp nút có đường đi.')

        # Chọn nút bắt đầu và nút đích từ danh sách các cặp nút có đường đi
        st.subheader('Chọn cặp nút để tìm kiếm đường đi')
        selected_pair = st.selectbox('Chọn cặp nút', reachable_pairs)

        if selected_pair:
            start_node, goal_node = selected_pair
            if st.button('Tìm kiếm'):
                if start_node and goal_node and start_node != goal_node:
                    if selected_algorithm == 'A*':
                        path = a_star_search(knowledge_tree, pos, start_node, goal_node)
                    elif selected_algorithm == 'Greedy Best-First Search':
                        path = gbfs_search(knowledge_tree, pos, start_node, goal_node)
                    elif selected_algorithm == 'Beam Search':
                        beam_width = 2  # Sử dụng giá trị mặc định hoặc thêm nhập liệu từ người dùng
                        path = beam_search(knowledge_tree, pos, start_node, goal_node, beam_width)
                    if path:
                        st.write(f'Đường đi từ {start_node} đến {goal_node}:', path)

                        # Lọc cây chỉ hiển thị các nhánh liên quan đến đường đi
                        filtered_tree = filter_subgraph_around_path(knowledge_tree, path)
                        plot_interactive_tree(filtered_tree, pos, layout, display_mode, path)

                        # Lưu đường đi vào CSV
                        paths = [(start_node, goal_node, " -> ".join(path))]
                        save_paths_to_csv(paths, "paths.csv")

                    else:
                        st.write(f'Không tìm thấy đường đi từ {start_node} đến {goal_node}')
                else:
                    st.warning('Hãy chọn các nút khác nhau để tìm kiếm.')
            else:
                st.warning('Hãy chọn cả nút bắt đầu và nút đích.')

    # Hiển thị các điểm chung giữa hai bộ dữ liệu
    st.subheader('Điểm chung giữa hai bộ dữ liệu')
    common_points = find_common_points(df_selected1, df_selected2, selected_columns)
    if common_points:
        save_and_plot_common_points(common_points)
    else:
        st.write('Không có điểm chung nào giữa hai bộ dữ liệu.')

    # Thêm các phân tích bổ sung
    st.subheader('Phân tích và Trực quan hóa bổ sung')

    analysis_option = st.selectbox('Chọn phân tích bổ sung', [
        'Phân tích tập trung', 
        'Phát hiện cộng đồng (Louvain)', 
        'Phát hiện cộng đồng (Girvan-Newman)', 
        'Phép đo mạng'])

    node_colors = None
    node_sizes = None

    if analysis_option == 'Phân tích tập trung':
        centrality_df = centrality_analysis(knowledge_tree)
        node_sizes = centrality_df.set_index('Node')['Degree Centrality'].to_dict()

    if analysis_option == 'Phát hiện cộng đồng (Louvain)':
        community_df = community_detection(knowledge_tree)
        node_colors = community_df.set_index('Node')['Community'].to_dict()

    if analysis_option == 'Phát hiện cộng đồng (Girvan-Newman)':
        community_df = community_detection_girvan_newman(knowledge_tree)
        node_colors = community_df.set_index('Node')['Community'].to_dict()

    if analysis_option == 'Phép đo mạng':
        metrics_df = network_metrics(knowledge_tree)

    # Trực quan hóa lại cây tri thức với các phân tích
    plot_interactive_tree(knowledge_tree, pos, layout, display_mode, node_colors=node_colors, node_sizes=node_sizes)
