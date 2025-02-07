import numpy as np
def Schwarz_Primitive(nx, ny, nz, domain):
    return np.cos(nx * domain[0]) + np.cos(ny * domain[1]) + np.cos(nz * domain[2])

def Schwarz_Diamond(nx, ny, nz, domain):
    return np.sin(nx * domain[0]) * np.sin(ny * domain[1]) * np.sin(nz * domain[2]) + np.sin(nx * domain[0]) * np.cos(ny * domain[1]) * np.cos(
                nz * domain[2]) + np.cos(nx * domain[0]) * np.sin(ny * domain[1]) * np.cos(nz * domain[2]) + np.cos(nx * domain[0]) * np.cos(
                ny * domain[1]) * np.sin(nz * domain[2])

def Shoen_Gyroid(nx, ny, nz, domain):
    return np.cos(nx * domain[0]) * np.sin(ny * domain[1]) + np.cos(ny * domain[1]) * np.sin(nz * domain[2]) + np.cos(nz * domain[2]) * np.sin(nx * domain[0])

def Neovius(nx, ny, nz, domain):
    return 3.0 * (np.cos(nx * domain[0]) + np.cos(ny * domain[1]) + np.cos(nz * domain[2])) + 4 * np.cos(nx * domain[0]) * np.cos(
                ny * domain[1]) * np.cos(nz * domain[2])

def iWP(nx, ny, nz, domain):
    return 2.0 * (np.cos(nx * domain[0]) * np.cos(ny * domain[1]) + np.cos(nz * domain[2]) * np.cos(nx * domain[0]) + np.cos(ny * domain[1]) * np.cos(
                nz * domain[2])) - (np.cos(2 * nx * domain[0]) + np.cos(2 * ny * domain[1]) + np.cos(2 * nz * domain[2]))

def P_W_Hybrid(nx, ny, nz, domain):
    return 4.0 * (np.cos(nx * domain[0]) * np.cos(ny * domain[1]) + np.cos(ny * domain[1]) * np.cos(nz * domain[2]) + np.cos(nz * domain[2]) * np.cos(
                nx * domain[0])) - 3 * np.cos(nx * domain[0]) * np.cos(ny * domain[1]) * np.cos(nz * domain[2]) + 2.4

def Lilinoid(nx, ny, nz, domain):
    return 0.5 * (np.sin(2 * nx * domain[0]) * np.cos(ny * domain[1]) * np.sin(nz * domain[2]) + np.sin(2 * ny * domain[1]) * np.cos(
                nz * domain[2]) * np.sin(nx * domain[0]) + np.sin(
                2 * nz * domain[2]) * np.cos(nx * domain[0]) * np.sin(ny * domain[1])) - (
                        1 / 2) * (np.cos(2 * nx * domain[0]) * np.cos(2 * ny * domain[1]) + np.cos(2 * ny * domain[1]) * np.cos(2 * nz * domain[2]) + np.cos(
                    2 * nz * domain[2]) * np.cos(2 * nx * domain[0])) + 0.15

def FKS(nx, ny, nz, domain):
    return np.cos(2 * nx * domain[0]) * np.sin(ny * domain[1]) * np.cos(nz * domain[2]) + np.cos(2 * ny * domain[1]) * np.sin(nz * domain[2]) * np.cos(
                nx * domain[0]) + np.cos(2 * nz * domain[2]) * np.sin(nx * domain[0]) * np.cos(ny * domain[1])

def Split_P(nx, ny, nz, domain):
    return 1.1 * (np.sin(2 * nx * domain[0]) * np.sin(nz * domain[2]) * np.cos(ny * domain[1]) + np.sin(2 * ny * domain[1]) * np.sin(
                nx * domain[0]) * np.cos(nz * domain[2]) + np.sin(
                2 * nz * domain[2]) * np.sin(ny * domain[1]) * np.cos(nx * domain[0])) - 0.2 * (
                        np.cos(2 * nx * domain[0]) * np.cos(2 * ny * domain[1]) + np.cos(2 * ny * domain[1]) * np.cos(2 * nz * domain[2]) + np.cos(
                    2 * nz * domain[2]) * np.cos(
                    2 * nx * domain[0])) - 0.4 * (np.cos(2 * nx * domain[0]) + np.cos(2 * ny * domain[1]) + np.cos(2 * nz * domain[2]))

def Shoen_FRD(nx, ny, nz, domain):
    return 4*np.cos(nx*domain[0])*np.cos(ny*domain[1])*np.cos(nz*domain[2])-(
            np.cos(2*nx*domain[0])*np.cos(2*ny*domain[1])+np.cos(2*nx*domain[0])*np.cos(2*nz*domain[2])+np.cos(2*ny*domain[1])*np.cos(2*nz*domain[2]))

"""Feel free to add more Surfaces and the respective name indexer to the dictionary.

Avoid special characters like - and _"""

surfaces_dict = {'Schwarz Primitive (SP)': Schwarz_Primitive,
                 'Schwarz Diamond (SD)': Schwarz_Diamond,
                 'Shoen Gyroid (SG)': Shoen_Gyroid,
                 'Neovius': Neovius,
                 'iWP': iWP,
                 'PWHybrid (PWH)': P_W_Hybrid,
                 'Lilinoid': Lilinoid,
                 'Fisher Kock S (FKS)': FKS,
                 'SplitP': Split_P,
                 'FRD': Shoen_FRD}


