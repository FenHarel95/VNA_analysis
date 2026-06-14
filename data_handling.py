import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, FloatSlider, IntSlider

class Analysis:
    """ Main class that handles analysis of VNA spectroscopy data.
    """
    def __init__(self, address, file_name:str, sample:str, setup:str, geo:str, **kwargs):
        self.address = address
        self.sample = sample
        self.file = address/file_name
        self.setup = setup
        self.geo = geo #Can be inP, outP
        self.components = ["11R", "11I", "12R", "12I", "21R", "21I", "22R", "22I"]
        self.user_ref = False
        #Addreses
        self.calc_add = "/calc/"
        self.raw_data_add = "/data/"
        self.info_add = "/info/"

        ### Custom variables according to setup
        if self.setup == "Konstanz_PSWS":
            self.rawS_add = "/data/PNA5225b "
            self.pow_add = "/data/PNA5225b power"
            self.frq_add = "/data/PNA5225b f"
            self.curr_add = "/data/magnet current easyd"
        if self.setup == "FZU_FMR":
            self.rawS_add = "data/VNA "
            self.pow_add = "/data/VNA power" #dBm
            self.frq_add = "/data/VNA frequency" #Hz
            self.curr_add = "/data/magnet current" #A
            self.field_add = "/data/magnet field" #T
        if self.setup == "CHAOS":
            self.rawS_add = "/data/PNA5225b "
            self.pow_add = "/data/PNA5225b power"
            self.frq_add = "/data/PNA5225b f"
            self.field_add = ""

        with h5py.File(self.file, "r") as f:
            self.power = f[self.pow_add][0]
            self.freqs = f[self.frq_add][0, :]*(1e-9) #GHz (Data should be saved in Hz)
            if self.setup in ("Konstanz_PSWS", "FZU_FMR"):
                # NumPy arrays
                self.curr = f[self.curr_add][:] #A (Data should be saved in A)
                self.field = self.calc_field(geo)
            else:
                self.field = f[self.field_add][:] #A (Data should be saved in T)
    
    def rawS_0(self, target):
        """Target must be of the type ijR or ijI"""
        with h5py.File(self.file, "r") as f:
            #NumPy arrays
            array = f[self.rawS_add + f"S{target}"][:]
        return array

    def get_ij(self, typ, target, add):
        """typ: S, Z, dL, etc. Target: ij. add: generic address to find them in h5 file"""
        with h5py.File(self.file, "r") as f:
            #NumPy arrays
            array = f[add + f"{typ}{target}"][:]
        return array

    def get_ij_indx(self, typ, target, indx, add):
        """typ: S, Z, dL, etc. Target: ij. indx (normally field) to extract, add: generic address to find them in h5 file"""
        with h5py.File(self.file, "r") as f:
            #NumPy arrays
            array = f[add + f"{typ}{target}"][indx,]
        return array

    def rawS_component(self, target):
        """Return the target part of raw S matrix"""
        array = self.get_ij("S", target, self.rawS_add)
        return array

    def calc_mag(self, re, im):
        """Magnitude of the complex number"""
        return np.sqrt(re*re+im*im)

    def calc_field(self, inP):
        """Calculates field in T, using a cal. below and experimental current
        inP only for FZU lab
        """
        A = self.curr#A
        if self.setup == "Konstanz_PSWS":
            field = 0.00776 + A * 0.02894 - A * A * 2.66123e-4 - A * A * A * 2.47495e-5 - A * A * A * A * 4.64487e-5
        if self.setup == "FZU_FMR":
            #Field for 20mmGap, from Manual (inPlane)
            #field1 = 0.01018 + 0.05859*A - 0.00118*(A**2) + 2.1658E-4*(A**3) - 1.77652E-5*(A**4) + 6.35589E-7*(A**5)
            #-1.15951E-8*(A**6) + 1.06504E-10*(A**7) - 3.91644E-13*(A**8)
            #Cal_12152025_Using_ml20240308b2_dS11_5dBm (inPlane)
            #field2 = 0.00675 + 0.05052*A + 6.69033E-4*(A**2) - 4.38998E-4*(A**3) + 1.11737E-4*(A**4) - 1.45078E-5*(A**5)
            #+ 1.00456E-6*(A**6) - 3.52383E-8*(A**7) + 4.88931E-10*(A**8)
            # field_inP = (field1 + field2)/2
            #Cal_18122025_Using_ml20240308b2_dS11_0dBm (outPlane)
            field_outP = (-4.75612131323894e-10*A**8 + 5.53240901595132e-8*A**7 - 2.44397032127303e-6*A**6 +
                          5.44573750852832e-5*A**5 - 0.000670050456046096*A**4 + 0.00455904626621348*A**3 -
                          0.0156901329345105*A**2 + 0.0762165481027225*A + 0.0164075279665365)
            #Measured 09/06/2026
            field_inP_mT = (-2.791826973437297e-12 * A ** 13 + 2.629557158787310e-10 * A ** 12 -
                            3.341952544859737e-09 * A ** 11 - 5.932640840884606e-07 * A ** 10 +
                            3.802178264636290e-05 * A ** 9 - 1.130650994442105e-03 * A ** 8 +
                            1.996752774903699e-02 * A ** 7 - 2.220353889564056e-01 * A ** 6 +
                            1.557372478377703e+00 * A ** 5 - 6.643370685736379e+00 * A ** 4 +
                            1.587659937311382e+01 * A ** 3 - 1.851724704466135e+01 * A ** 2 +
                            5.928607446113774e+01 * A + 6.050126767279399e+00)
            field_inP = field_inP_mT / 1000  # T

            if inP:
                field = field_inP
            else:
                field = field_outP
        if self.setup == "CHAOS":
            field = self.field
        return field

    def plot_ij_plotly(self, dic, idx:int, typ:str, low_x, high_x, comment:str, save = True, plot = True):
        """Plots dic_ij matrix. Assumes dic contains all ijR and ijI components.
        typ: is the type of quantity, e.g. S, Z, dZ, dL, etc."""
        freqs_ghz = self.freqs  # assumes GHz

        # Create figure with two subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1)

        def get_lines(n):
            m_11 = self.calc_mag(dic[typ+'11R'][n], dic[typ+'11I'][n])
            m_12 = self.calc_mag(dic[typ+'12R'][n], dic[typ+'12I'][n])
            m_21 = self.calc_mag(dic[typ+'21R'][n], dic[typ+'21I'][n])
            m_22 = self.calc_mag(dic[typ+'22R'][n], dic[typ+'22I'][n])
            lines_o = [
                {'y': dic[typ+'11R'][n], 'name': f'Re({typ}11)', 'color': 'blue', 'row': 1},
                {'y': dic[typ+'11I'][n], 'name': f'Im({typ}11)', 'color': 'red', 'row': 1},
                {'y': dic[typ+'22R'][n], 'name': f'Re({typ}22)', 'color': 'cyan', 'row': 1},
                {'y': dic[typ+'22I'][n], 'name': f'Im({typ}22)', 'color': 'magenta', 'row': 1},
                {'y': m_11, 'name': f'Mag({typ}11)', 'color': 'orange', 'row': 1},
                {'y': m_22, 'name': f'Mag({typ}22)', 'color': 'mediumpurple', 'row': 1},

                {'y': dic[typ+'12R'][n], 'name': f'Re({typ}12)', 'color': 'blue', 'row': 2},
                {'y': dic[typ+'12I'][n], 'name': f'Im({typ}12)', 'color': 'red', 'row': 2},
                {'y': dic[typ+'21R'][n], 'name': f'Re({typ}21)', 'color': 'cyan', 'dash': 'dash', 'row': 2},
                {'y': dic[typ+'21I'][n], 'name': f'Im({typ}21)', 'color': 'magenta', 'dash': 'dash', 'row': 2},
                {'y': m_12, 'name': f'Mag({typ}12)', 'color': 'orange', 'row': 2},
                {'y': m_21, 'name': f'Mag({typ}21)', 'color': 'mediumpurple', 'dash': 'dash', 'row': 2}
            ]
            return lines_o

        lines = get_lines(idx)
        for line in lines:
            fig.add_trace(go.Scattergl(
                x=freqs_ghz,
                y=line['y'],
                mode='lines',
                name=line['name'],
                line=dict(color=line['color'], dash=line.get('dash', 'solid'))
            ), row=line['row'], col=1)

        # Update layout
        fig.update_layout(
            height=800,
            width=850,
            title=f"Plotting {typ}ij for H({idx})=" + str(int(1000 * self.field[idx])) + "mT",
            showlegend=True,
            xaxis=dict(title="frequency (GHz)", range=[low_x, high_x]),
            xaxis2=dict(title="frequency (GHz)", range=[low_x, high_x]),
            yaxis=dict(title=f"{typ}ii (pH)", range=[-500, 500]),
            yaxis2=dict(title=f"{typ}ij (pH)", range=[-150, 150]),
        )

        fig.update_xaxes(title_font_size=22, tickfont_size=16)
        fig.update_yaxes(title_font_size=22, tickfont_size=16)

        if save:
            fig.write_html(self.address / (self.sample + f"_{typ}ij_vsf" + comment + f"_{int(self.power)}dBm.html"))

        if plot:
            fig.show()


    def dic_typ(self, typ, add):
        """Returns a dictionary with the ij matrix of choise:typ from the addres:add"""
        data_dict = {(typ+name): self.get_ij(typ, name, add) for name in self.components}
        data_dict = {k: v.astype(np.float32) for k, v in data_dict.items()}  # ensuring compressed data to 4 bits
        return data_dict

    def dic_typ_indx(self, typ, indx, add):
        """Returns a dictionary with the ij matrix of choise:typ from the addres:add"""
        data_dict = {(typ+name): self.get_ij_indx(typ, name, indx, add) for name in self.components}
        data_dict = {k: v.astype(np.float32) for k, v in data_dict.items()}  # ensuring compressed data to 4 bits
        return data_dict

    def delta_dic_typ(self, typ1, add1, typ2, add2, sign=-1):
        """Returns a dictionary with the ij matrix of choise:typ from the addres:add"""
        data_dict = {(typ1+name): (self.get_ij(typ1, name, add1) + sign*self.get_ij(typ2, name, add2) ) for name in self.components}
        data_dict = {k: v.astype(np.float32) for k, v in data_dict.items()}  # ensuring compressed data to 4 bits
        return data_dict

    def subtract_background_ij(self, typ, target, add, single, ref_typ, ref_indx, ref_add, n_bg, sign=-1):
        initial = self.get_ij(typ, target, add)
        if single:
            background = self.get_ij_indx(ref_typ, target, ref_indx, ref_add)
        else:
            # Background components method
            U, S, Vt = np.linalg.svd(initial, full_matrices=False)
            #background = np.outer(U[:, 0] * S[0], Vt[0, :])  # background = first component
            background = (
                    U[:, :n_bg]
                    @ np.diag(S[:n_bg])
                    @ Vt[:n_bg, :]
            )
            #initial = initial + sign*background
            #background = np.zeros(len(initial[0]), dtype=complex)

        final = initial + sign*background
        return final, background #final has a shape [field,freqs], background [freqs]

    def subtract_background_typ(self, typ, add, single, ref_indx, ref_add, n_bg, sign=-1):
        data_dict = {}
        backg_dict = {}
        for name in self.components:
            data, backg = self.subtract_background_ij(typ, name, add, single, typ, ref_indx, ref_add, n_bg, sign)
            data_dict["d_"+typ+name] = data
            backg_dict["ref_"+typ+name] = backg
        data_dict = {k: v.astype(np.float32) for k, v in data_dict.items()}  # ensuring compressed data to 4 bits
        backg_dict = {k: v.astype(np.float32) for k, v in backg_dict.items()} # ensuring compressed data to 4 bits
        return data_dict, backg_dict

    def plot_rawSij_plotly(self, idx, low_x, high_x, save=True, plot=True):
        """Plots the raw S matrix"""
        typ = "S"
        self.plot_ij_plotly(self.dic_typ(typ, self.rawS_add), idx, typ, low_x, high_x, "",save, plot)

    def write_ij_component(self, dataArray, address, key):
        with h5py.File(self.file, "a") as f:  # Open in append mode
            add = address + key
            if (add) in f:
                del f[add]  # Delete if it exists
            f.create_dataset(add, data=dataArray, compression="gzip", compression_opts=9)
        #print("Set saved in HDF5 file.")

    def store_ref_indx(self, ref_indx):
        self.ref_indx_info = ref_indx
        with h5py.File(self.file, "a") as f:
            if ("/info/Ref_idx") in f:
                del f["/info/Ref_idx"]  # Delete if it exists
            f.create_dataset("/info/Ref_idx", data=self.ref_indx_info)


class Analysis_FMR(Analysis):
    """ Main class that handles analysis of FMR spectroscopy data.
    """

    def __init__(self, address, file_name: str, sample: str, setup: str, geo: str, vna_ref:str, **kwargs):
        super().__init__(address, file_name, sample, setup, geo, **kwargs)
        self.sample = sample + "_" + geo
        self.vna_ref = vna_ref #Reference is already subtracted from data.This normally is the case up to 11/06/2026.
        self.ref = False
        self.ref_indx = self.read_ref_indx()
        self.rawS_add = "/data/ "

        if vna_ref is True:
            self.ref = True
            if self.ref_indx is None:
                self.original_data_add = "/data/VNA "
                self.original_ref_add = self.original_data_add

                rawS_dict = self.rawS() # calculate the raw data
                original_dS_matrix = self.dic_typ("d_S", self.original_data_add) #Get the original dS

                for key in rawS_dict:
                    self.write_ij_component(rawS_dict[key], self.rawS_add, key) # store it in the file

                for key in original_dS_matrix:
                    self.write_ij_component(original_dS_matrix[key], self.calc_add, key) # store it in the file

                self.store_ref_indx("VNA ref")
                self.ref_indx = None

        else:
            self.data_add = None
            if self.ref_indx is not None:
                self.ref = True


    def read_ref_indx(self):
        with h5py.File(self.file, "r") as f:
            if "/info/Ref_idx" in f:
                self.ref_indx_info = f["/info/Ref_idx"]
            else:
                self.ref_indx_info = None

        return self.ref_indx_info

    def rawS(self):
        """Return the raw S matrix"""
        if self.vna_ref:
            data_dict = {}
            for name in self.components:
                data, backg = self.subtract_background_ij("d_S", name, self.original_data_add, True,
                                                          "ref_S", 0, self.original_ref_add, sign=1)
                data_dict["S" + name] = data
            data_dict = {k: v.astype(np.float32) for k, v in
                         data_dict.items()}  # ensuring compressed data to 4 bits
            return data_dict
        else:
            dict = self.dic_typ("S", self.rawS_add)
        return dict

    def subtract_ref(self, single, nb_g=1, ref_indx= None):
        """Calculates the raw data (files with subtraction) or the subtraction of ref. (files of raw data)"""
        #if single: ########This block should not be here, because it is not real when single=False
        #
        #    ref_dict = self.dic_typ_indx("S", ref_indx, self.raw_add)
        #    for key in ref_dict:
        #        self.write_ij_component(ref_dict[key], self.ref_add, key)
        self.ref = True
        self.user_ref = True
        if single:
            self.ref_indx = ref_indx
            self.store_ref_indx(self.ref_indx)
        else:
            self.ref_indx = None
            self.store_ref_indx("AverageBackground")
        deltaS_dict, backg_dict = self.subtract_background_typ("S",
                                     self.rawS_add, single, self.ref_indx, self.rawS_add, nb_g, sign=-1)
        for key in deltaS_dict:
            self.write_ij_component(deltaS_dict[key], self.calc_add, key)
        for key in backg_dict:
            self.write_ij_component(backg_dict[key], self.calc_add, key)

    def d_S(self, target):
        """Return the dSij array"""
        array = self.get_ij("d_S", target, self.calc_add)
        return array

    def plot_dij_plotly(self, typ, idx, low_x, high_x, save=False, plot=True):
        """Plots the "typ"ij matrix for data stored in self.cal_add. e.g.: dSij matrix"""

        if self.ref:
            comment = f"Ref_index:{self.read_ref_indx()}"
            self.plot_ij_plotly(self.dic_typ(typ, self.calc_add), idx, typ, low_x, high_x, comment, save, plot)
        else:
            print("No ref. subtracted yet. Nothing to show.")

    def plot_dij_slider(self, typ, n_0):
        n_max = (self.get_ij("S", "11R", self.rawS_add)).shape[0] - 1

        # Interactive sliders
        def plot_dij(index):
            self.plot_dij_plotly(typ, index, self.freqs[0], np.max(self.freqs), save=False, plot=True)

        interact(plot_dij,
                 index=IntSlider(value=n_0, min=0, max=n_max, step=1, readout_format='.0f', description=r"Field index")
                 );

class Analysis_PSWS(Analysis):
    """ Main class that handles analysis of PSWS spectroscopy data.
    """
    def __init__(self, address , file_name:str, sample:str , setup:str, geo:str, device: str, **kwargs):
        super().__init__(address, file_name, sample, setup, geo, **kwargs)
        self.sample = sample + "_" + device
        self.calc_data_add = "/calc/"
        with h5py.File(self.file, "r") as f:
            self.ref_idx = f["/info/Ref_idx"][()]  # [()] reads the scalar value
        if self.ref_idx == -1:
            self.ref = False
        else:
            self.ref = True

    def dL(self, target):
        """Return the dLij array"""
        array = self.get_ij("dL", target, self.calc_data_add)
        return array

    def dS(self, target):
        """Return the dLij array"""
        array = self.get_ij("dS", target, self.calc_data_add)
        return array

    def plot_dij_plotly(self, typ, idx, low_x, high_x, save=True, plot=True):
        """Plots the "typ"ij matrix for data stored in self.calc_data_add"""
        if self.ref:
            comment = f"_Refi_{self.ref_idx}"
        else:
            comment = "_MeanRef"
        self.plot_ij_plotly(self.dic_ij(typ, self.calc_data_add), idx, typ, low_x, high_x, comment, save, plot)

    def plot_dij_slider(self, typ, n_0):
        n_max = (self.rawS("11R")).shape[0] - 1

        # Interactive sliders
        def plot_dij(index):
            self.plot_dij_plotly(typ, index, self.freqs[0], np.max(self.freqs), save=False, plot=True)

        interact(plot_dij,
                 index=IntSlider(value=n_0, min=0, max=n_max, step=1, readout_format='.0f', description=r"Field index")
                 );


class FitStore:
    def __init__(self, filename, group="fit_results"):
        self.filename = filename
        self.group = group

    # -------------------------
    # initialization
    # -------------------------
    def _init(self, grp, fit_dict):

        grp.create_dataset("index", shape=(0,), maxshape=(None,), dtype="i8")
        grp.create_dataset("field", shape=(0,), maxshape=(None,), dtype="f8")

        for name in fit_dict["params"]:
            grp.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")
            grp.create_dataset(name + "_err", shape=(0,), maxshape=(None,), dtype="f8")

    # -------------------------
    # main save function
    # -------------------------
    def save(self, index, field, fit_dict):

        with h5py.File(self.filename, "a") as f:

            grp = f.require_group(self.group)

            if "index" not in grp:
                self._init(grp, fit_dict)

            indices = grp["index"][:]

            # -------------------------------------------------
            # 1. CHECK EXISTING ENTRY (ONLY BY INDEX)
            # -------------------------------------------------
            match = np.where(indices == index)[0]

            if len(match) > 0:
                row = int(match[0])
                overwrite = True
            else:
                row = len(indices)
                overwrite = False

                # expand datasets
                for name in grp.keys():
                    grp[name].resize((row + 1,))

                grp["index"][row] = index

            # -------------------------------------------------
            # 2. WRITE METADATA (field is NEVER used for logic)
            # -------------------------------------------------
            grp["field"][row] = field

            # -------------------------------------------------
            # 3. WRITE PARAMETERS
            # -------------------------------------------------
            for name, val in fit_dict["params"].items():
                grp[name][row] = val

                err = fit_dict["errors"].get(name, None)
                grp[name + "_err"][row] = err if err is not None else np.nan