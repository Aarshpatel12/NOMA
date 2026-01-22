import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance_matrix

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="VLC Beam Steering & NOMA", layout="wide")
st.title("🔦 Smart VLC: Beam Steering, Clustering & NOMA")
st.markdown("""
This app simulates **VLC User Clustering (VUC)** and **Power Allocation**. 
It demonstrates how users are grouped and how NOMA assigns power based on distance.
""")

# Sidebar Controls
st.sidebar.header("Simulation Settings")
num_users = st.sidebar.slider("Number of Users", 2, 20, 10)
num_beams = st.sidebar.slider("Number of Beams (Clusters)", 1, 5, 3)
room_size = st.sidebar.slider("Room Size (meters)", 4, 12, 6)
show_hull = st.sidebar.checkbox("Show Search Space (Convex Hull)", value=True)
show_noma = st.sidebar.checkbox("Show NOMA Pairings (Strong/Weak)", value=True)

# --- 2. CORE ALGORITHMS ---

def generate_users(n, size):
    """Generate random user locations in the room."""
    return np.random.rand(n, 2) * size

def get_convex_hull(points):
    """Calculates the convex hull for search space reduction."""
    if len(points) < 3: 
        return None # Convex hull requires at least 3 points
    hull = ConvexHull(points)
    return points[hull.vertices]

def run_vuc_step(users, beams, assignments):
    """
    Simulates one iteration of the VUC Algorithm.
    1. Update Beam Centers (Optimization Step) - Simplified as centroid
    2. Re-assign Users to closest beam (Assignment Step)
    """
    # Step 1: Update Beam positions
    new_beams = beams.copy()
    for i in range(len(beams)):
        cluster_points = users[assignments == i]
        if len(cluster_points) > 0:
            new_beams[i] = np.mean(cluster_points, axis=0)
    
    # Step 2: Assign users to nearest beam
    dists = distance_matrix(users, new_beams)
    new_assignments = np.argmin(dists, axis=1)
    
    return new_beams, new_assignments

# --- 3. SESSION STATE MANAGEMENT ---
if 'users' not in st.session_state:
    st.session_state.users = generate_users(num_users, room_size)
if 'beams' not in st.session_state:
    st.session_state.beams = generate_users(num_beams, room_size)
if 'assignments' not in st.session_state:
    st.session_state.assignments = np.zeros(num_users, dtype=int)

# Reset button logic
if st.sidebar.button("Reset Simulation"):
    st.session_state.users = generate_users(num_users, room_size)
    st.session_state.beams = generate_users(num_beams, room_size)
    st.session_state.assignments = np.zeros(num_users, dtype=int)

# --- 4. VISUALIZATION & LOGIC ---

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, room_size)
    ax.set_ylim(0, room_size)
    ax.set_title(f"Room View ({room_size}x{room_size}m)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, linestyle='--', alpha=0.6)

    colors = ['#FF4B4B', '#1E90FF', '#2E8B57', '#FFD700', '#9370DB'] 
    
    users = st.session_state.users
    beams = st.session_state.beams
    assignments = st.session_state.assignments

    # Draw Data
    for i, beam in enumerate(beams):
        color = colors[i % len(colors)]
        cluster_users_idx = np.where(assignments == i)[0]
        cluster_users = users[cluster_users_idx]

        # Draw Beam Center
        ax.scatter(beam[0], beam[1], color=color, s=300, marker='X', edgecolors='black', label=f"Beam {i+1}")

        # Draw Users and Connections
        for u_idx, user_pos in zip(cluster_users_idx, cluster_users):
            ax.plot([user_pos[0], beam[0]], [user_pos[1], beam[1]], color=color, alpha=0.3)
            ax.scatter(user_pos[0], user_pos[1], color=color, s=100)

        # Draw Convex Hull (Search Space)
        if show_hull and len(cluster_users) >= 3:
            hull_points = get_convex_hull(cluster_users)
            if hull_points is not None:
                hull_points = np.vstack((hull_points, hull_points[0]))
                ax.plot(hull_points[:,0], hull_points[:,1], color=color, linestyle='--', linewidth=2, alpha=0.8)
                ax.fill(hull_points[:,0], hull_points[:,1], color=color, alpha=0.1)

    ax.legend(loc='upper right')
    st.pyplot(fig)

    # --- ADDED SECTION: Challenges Text ---
    with st.expander("ℹ️ Challenges in NOMA & Power Allocation (Read More)", expanded=True):
        st.markdown("""
        ### **1. Error propagation in SIC**
        * Accurate **channel estimation** use karte hain
        * **User pairing** carefully karte hain (strong + weak user)
        * Error kam karne ke liye **improved SIC algorithms** use kiye jaate hain

        ### **2. Complexity in power allocation**
        * **Simple power allocation rules** (fixed or ratio-based) use kiye jaate hain
        * **Optimization algorithms** ya **machine learning** ka use karke power decide ki jaati hai
        * Users ko **clusters** mein divide karke complexity kam ki jaati hai
        """)

with col2:
    st.subheader("Simulation Control")
    st.write("Click 'Step' to optimize clusters.")
    
    if st.button("Step (Iterate)"):
        new_beams, new_assignments = run_vuc_step(users, beams, assignments)
        st.session_state.beams = new_beams
        st.session_state.assignments = new_assignments
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Metrics & NOMA")
    st.write(f"**Total Users:** {num_users}")
    
    # Calculate Search Space Reduction
    if show_hull:
        total_area = room_size ** 2
        hull_area = 0
        for i in range(num_beams):
            cluster_users = users[assignments == i]
            if len(cluster_users) >= 3:
                hull = ConvexHull(cluster_users)
                hull_area += hull.volume
        reduction = (1 - (hull_area / total_area)) * 100
        st.success(f"Search Space Reduced: {reduction:.1f}%")

    # --- ADDED: NOMA Power Allocation Logic ---
    if show_noma:
        st.markdown("#### ⚡ Power Allocation (NOMA)")
        st.caption("Auto-detects Weak vs Strong users based on distance.")
        
        for i in range(num_beams):
            # Find users in this cluster
            cluster_indices = np.where(assignments == i)[0]
            if len(cluster_indices) > 0:
                beam_pos = beams[i]
                # Calculate distances
                dists = [np.linalg.norm(users[idx] - beam_pos) for idx in cluster_indices]
                
                # Sort: Furthest (Weak) to Closest (Strong)
                # In NOMA, Furthest gets High Power, Closest gets Low Power
                sorted_pairs = sorted(zip(cluster_indices, dists), key=lambda x: x[1], reverse=True)
                
                with st.expander(f"Beam {i+1} Power Stats", expanded=False):
                    for rank, (u_idx, d) in enumerate(sorted_pairs):
                        # Simple logic: Top 50% furthest are 'Weak', rest 'Strong'
                        is_weak = rank < len(sorted_pairs) / 2
                        role = "🔴 Weak (High Power)" if is_weak else "🟢 Strong (Low Power)"
                        st.write(f"**User {u_idx+1}:** {d:.2f}m → {role}")