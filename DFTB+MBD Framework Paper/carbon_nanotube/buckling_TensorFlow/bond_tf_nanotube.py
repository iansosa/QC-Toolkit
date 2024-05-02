import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def Ecov_nanotube(coords,bond_id,r0,kr,hessian=False):
    sess = tf.Session()
    coords = tf.constant(coords, dtype=tf.float64)
    E_list = []
    for id in bond_id:
        if id[0] < id[1]:
            r_temp = tf.sqrt(tf.reduce_sum((coords[id[0]]-coords[id[1]])**2))
            E_list.append(0.5*kr*(r_temp-r0)**2)
    E = tf.reduce_sum(E_list)
    if hessian:
        K = tf.hessians(E, coords)[0]
        return sess.run(K)
    else:
        F = tf.gradients(E, coords)[0]
        return sess.run(E), sess.run(F)

def Ecov_nanotube_assemble(coords,bond_id,r0_list,kr,hessian=False):
    # Initialize E, F, K
    F = np.zeros([len(coords),3])
    K = np.zeros([len(coords),3,len(coords),3])
    E_list = []

    # Variables
    coords01 = tf.placeholder(tf.float64, shape=(2,3))
    r0 = tf.placeholder(tf.float64, shape=(1,))
    # covalent bond
    r_temp = tf.sqrt(tf.reduce_sum((coords01[0] - coords01[1]) ** 2))
    E_temp = 0.5 * kr * (r_temp - r0) ** 2
    F_temp = tf.gradients(E_temp, coords01)

    if hessian:
        K_temp = tf.hessians(E_temp, coords01)

    # loop over all the bonds and assemble them to matrices
    with tf.Session() as sess:
        for i_bond in range(len(bond_id)):
            id = bond_id[i_bond]
            r0_i = r0_list[i_bond]
            if id[0] < id[1]:
                feed_dict = {coords01:[coords[id[0]],coords[id[1]]],r0:[r0_i]}
                if hessian:
                    K_fed = sess.run(K_temp,feed_dict=feed_dict)[0]
                    for i in range(2):
                        for j in range(2):
                            K[id[i],:,id[j],:] += K_fed[i,:,j,:]

                else:
                    E_list.append(sess.run(E_temp,feed_dict=feed_dict))
                    F_fed = sess.run(F_temp,feed_dict=feed_dict)[0]
                    for i in range(2):
                        F[id[i]] += F_fed[i]

    if hessian:
        return K
    else:
        return sum(E_list), F

def Ebend_nanotube(coords,bend_id,theta0,k_theta,hessian=False):
    sess = tf.Session()
    coords = tf.constant(coords, dtype=tf.float64)
    E_list = []
    for id in bend_id:
        a = coords[id[0]]
        b = coords[id[1]]
        c = coords[id[2]]
        ab = b - a
        ac = c - a
        theta = tf.math.acos(tf.reduce_sum(ab*ac)/(tf.norm(ab)*tf.norm(ac)))
        E_list.append(0.5*k_theta*(theta-theta0)**2)
    E = tf.reduce_sum(E_list)
    if hessian:
        K = tf.hessians(E, coords)[0]
        return sess.run(K)
    else:
        F = tf.gradients(E, coords)[0]
        return sess.run(E), sess.run(F)

def Ebend_nanotube_assemble(coords,bend_id,theta0_list,k_theta,hessian=False):
    # Initialize E, F, K
    F = np.zeros([len(coords),3])
    K = np.zeros([len(coords),3,len(coords),3])
    E_list = []

    # Variables
    coords012 = tf.placeholder(tf.float64, shape=(3, 3))
    theta0 = tf.placeholder(tf.float64, shape=(1,))

    #
    ab = coords012[1] - coords012[0]
    ac = coords012[2] - coords012[0]
    theta = tf.math.acos(tf.tensordot(ab,ac,1)/(tf.norm(ab)*tf.norm(ac)))
    E_temp = 0.5*k_theta*(theta-theta0)**2
    F_temp = tf.gradients(E_temp, coords012)

    if hessian:
        K_temp = tf.hessians(E_temp,coords012)

    # loop over all the bonds and assemble them to matrices
    with tf.Session() as sess:
        for i_bend in range(len(bend_id)):
            id = bend_id[i_bend]
            theta0_i = theta0_list[i_bend]
            feed_dict = {coords012:[coords[id[0]],coords[id[1]],coords[id[2]]],
                         theta0:[theta0_i]}
            if hessian:
                K_fed = sess.run(K_temp,feed_dict=feed_dict)[0]
                for i in range(3):
                    for j in range(3):
                        K[id[i],:,id[j],:] += K_fed[i,:,j,:]

            else:
                E_list.append(sess.run(E_temp,feed_dict=feed_dict))
                F_fed = sess.run(F_temp,feed_dict=feed_dict)[0]
                for i in range(3):
                    F[id[i]] += F_fed[i]

    if hessian:
        return K
    else:
        return sum(E_list), F

def neighbour_search(bond_id):
    neighbour_list = []
    for id in bond_id:
        if id[0] + 1 > len(neighbour_list):
            neighbour_list.append([])
        neighbour_list[id[0]].append(id[1])
    return neighbour_list

def Edihedral_nanotube(coords,bond_id,neighbour,phi0_list,k_phi,hessian=False):
    # Initialize E, F, K
    F = np.zeros([len(coords), 3])
    K = np.zeros([len(coords), 3, len(coords), 3])
    E_list = []

    #
    if np.size(np.array(phi0_list)) == 1:
        phi0_list = phi0_list * np.ones([3664,1])

    # Variables
    coords0123 = tf.placeholder(tf.float64, shape=(4, 3))
    phi0 = tf.placeholder(tf.float64, shape=(1,))

    u1 = coords0123[1] - coords0123[0]
    u2 = coords0123[2] - coords0123[1]
    u3 = coords0123[3] - coords0123[2]
    atan2_y = tf.norm(u2) * tf.tensordot(u1,tf.linalg.cross(u2,u3),1)
    atan2_x = tf.tensordot(tf.linalg.cross(u1, u2), tf.linalg.cross(u2, u3), 1)
    phi = tf.math.atan2(atan2_y,atan2_x)
    phi = tf.math.abs(phi)
    phi = tf.cond(phi > np.pi/2,lambda: tf.add(phi,-np.pi), lambda: tf.add(phi,0))
    E_temp = 0.5 * k_phi * (phi - phi0) ** 2
    F_temp = tf.gradients(E_temp, coords0123)

    if hessian:
        K_temp = tf.hessians(E_temp,coords0123)

    with tf.Session() as sess:
        count = 0
        for id in bond_id:
            if id[0] < id[1]:
                for id0 in neighbour[id[0]]:
                    for id1 in neighbour[id[1]]:
                        if id0 != id[1] and id1 != id[0]:
                            id_list = [id0,id[0],id[1],id1]
                            feed_dict = {coords0123: [coords[id0], coords[id[0]], coords[id[1]], coords[id1]],
                                         phi0: [phi0_list[count]]}
                            count += 1
                            if hessian:
                                K_fed = sess.run(K_temp, feed_dict=feed_dict)[0]
                                for i in range(4):
                                    for j in range(4):
                                        K[id_list[i], :, id_list[j], :] += K_fed[i, :, j, :]

                            else:
                                E_list.append(sess.run(E_temp, feed_dict=feed_dict))
                                F_fed = sess.run(F_temp, feed_dict=feed_dict)[0]
                                # print(sess.run(phi, feed_dict=feed_dict),id_list)
                                # print(sess.run(phi-phi0, feed_dict=feed_dict))
                                for i in range(4):
                                    F[id_list[i]] += F_fed[i]

    if hessian:
        return K
    else:
        return sum(E_list), F

def dihedral_measure(coords,bond_id,neighbour):
    dih = []
    # Variables
    coords0123 = tf.placeholder(tf.float64, shape=(4, 3))

    u1 = coords0123[1] - coords0123[0]
    u2 = coords0123[2] - coords0123[1]
    u3 = coords0123[3] - coords0123[2]
    atan2_y = tf.norm(u2) * tf.tensordot(u1,tf.linalg.cross(u2,u3),1)
    atan2_x = tf.tensordot(tf.linalg.cross(u1, u2), tf.linalg.cross(u2, u3), 1)
    phi = tf.math.atan2(atan2_y,atan2_x)
    phi = tf.math.abs(phi)
    phi = tf.cond(phi > np.pi/2,lambda: tf.add(phi,-np.pi), lambda: tf.add(phi,0))

    with tf.Session() as sess:
        for id in bond_id:
            if id[0] < id[1]:
                for id0 in neighbour[id[0]]:
                    for id1 in neighbour[id[1]]:
                        if id0 != id[1] and id1 != id[0]:
                            feed_dict = {coords0123: [coords[id0], coords[id[0]], coords[id[1]], coords[id1]]}
                            dih_temp = sess.run(phi, feed_dict=feed_dict)
                            dih.append(dih_temp)
    return dih

def bond_search(XX,YY,ZZ,r0,output_neighbour_list=False):
    bond_list = []
    neighbour_list = []
    Natm = len(XX)

    for i in range(Natm):
        neighbour_list_temp = []
        for j in range(Natm):
            dij = ((XX[i]-XX[j])**2+(YY[i]-YY[j])**2+(ZZ[i]-ZZ[j])**2)**0.5
            if 0 + 1e-13 < dij <= r0 and i != j:
                bond_list.append([int(i+1),int(j+1),dij])
                neighbour_list_temp.append(int(j+1))
        neighbour_list.append(neighbour_list_temp)
    if output_neighbour_list:
        return np.array(bond_list), neighbour_list
    else:
        return np.array(bond_list)

def bend_search(XX,YY,ZZ,r0):
    from itertools import combinations
    bond_list, neighbour_list = bond_search(XX,YY,ZZ, r0, output_neighbour_list=True)
    bend_list = []

    for i in range(len(neighbour_list)):
        combins = list(combinations(neighbour_list[i],2))
        for j in range(len(combins)):
            temp = [i+1]+list(combins[j])
            ab = np.array([XX[temp[1]-1]-XX[temp[0]-1],YY[temp[1]-1]-YY[temp[0]-1],ZZ[temp[1]-1]-ZZ[temp[0]-1]])
            ac = np.array([XX[temp[2]-1]-XX[temp[0]-1],YY[temp[2]-1]-YY[temp[0]-1],ZZ[temp[2]-1]-ZZ[temp[0]-1]])
            theta = np.math.acos(np.tensordot(ab, ac, 1) / (np.linalg.norm(ab) * np.linalg.norm(ac)))
            bend_list.append(temp+[theta])

    return np.array(bend_list)
