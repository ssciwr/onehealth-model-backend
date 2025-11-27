import logging

import numpy as np

from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
)
from heiplanet_models.Pmodel.Pmodel_rates_development import (
    mosq_dev_i,
    mosq_dev_j,
)
from heiplanet_models.Pmodel.Pmodel_rates_mortality import (
    mosq_mort_a,
    mosq_mort_e,
    mosq_surv_ed,
    mosq_mort_j,
)

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def eqsys(v, vars):
    # Unpack variables
    (
        t_idx,
        step_t,
        Temp,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = vars

    logger.debug(f"dimension v[...,4]: {v[...,4].shape}")
    logger.debug(f"\ttype: v[...,4]: {type(v[...,4])}")

    logger.debug(f"dimension birth: {birth.shape}")
    logger.debug(f"\ttype: birth: {type(birth)}")

    logger.debug(f"dimension dia_lay: {dia_lay.shape}")
    logger.debug(f"\ttype: dia_lay: {type(dia_lay)}")

    logger.debug(f"dimension mort_e: {mort_e.shape}")
    logger.debug(f"\ttype: mort_e: {type(mort_e)}")

    logger.debug(f"dimension water_hatch: {water_hatch.shape}")
    logger.debug(f"\ttype: water_hatch: {type(water_hatch)}")

    logger.debug(f"dimension v[...,0]: {v[...,0].shape}")
    logger.debug(f"\ttype: v[...,0]: {type(v[...,0])}")

    FT = np.zeros_like(v)

    logger.debug(f"dimension FT[...,0]: {FT[...,0].shape}")

    # Differential equations (vectorized over grid)
    # Egg compartment (non-diapause)
    FT[..., 0] = (
        # v[..., 4] * birth * (1 - dia_lay) - (mort_e + water_hatch * dev_e) * v[..., 0]
        v[..., 4].data * birth * (1-dia_lay) - (mort_e + (water_hatch * dev_e)) * v[..., 0].data
    )

    # Egg compartment (diapause)
    FT[..., 1] = (
        v[..., 4].data * birth * dia_lay - water_hatch * dia_hatch * v[..., 1].data  # Hatching from diapause
    )

    # Juvenile compartment
    FT[..., 2] = (
        water_hatch * dev_e * v[..., 0].values  # Hatching from non-diapause eggs
        + water_hatch * dia_hatch * ed_surv * v[..., 1].values  # Hatching from diapause eggs
        - (mort_j + dev_j) * v[..., 2].values  # Mortality and development
        - (v[..., 2].data ** 2) / CC[..., t_idx - 1]  # Density-dependent mortality
    )

    # Immature adult compartment
    FT[..., 3] = (
        0.5 * dev_j * v[..., 2]  # Development from juveniles
        #- (mort_a + dev_i) * v[..., 3]  # Mortality and maturation
    )

    # Mature adult compartment
    FT[..., 4] = (
        dev_i * v[..., 3]  # Maturation from immature adults
        #- mort_a * v[..., 4]  # Adult mortality
    )

    # Replace NaNs
    FT[np.isnan(-FT)] = -v.values[np.isnan(-FT)] * step_t
    return FT


def eqsys_log(v, vars):
    # Unpack variables
    (
        t_idx,
        step_t,
        Temp,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = vars

    FT = np.zeros_like(v)
    # Revisited
    FT[..., 0] = v[..., 4] * birth * (1 - dia_lay) / v[..., 0] - (
        mort_e + water_hatch * dev_e
    )

    # Revisited
    FT[..., 1] = v[..., 4] * birth * dia_lay / v[..., 1] - water_hatch * dia_hatch

    # Revisited
    FT[..., 2] = (
        water_hatch * dev_e * v[..., 0] / v[..., 2]
        + water_hatch * dia_hatch * ed_surv * v[..., 1] / v[..., 2]
        - (mort_j + dev_j)
        - v[..., 2] / CC[..., t_idx - 1]
    )

    # Revisited
    FT[..., 3] = 0.5 * dev_j * v[..., 2] / v[..., 3] - (mort_a + dev_i)

    # Revisited
    FT[..., 4] = dev_i * v[..., 3] / v[..., 4] - mort_a

    FT[np.isnan(-FT)] = -v[np.isnan(-FT)] * step_t
    return FT


def rk4_step(f, flog, v, vars, step_t):
    # Octave-style RK4 with negative value correction using log-form ODEs
    k1 = f(v, vars)
    k2 = f(v + 0.5 * k1 / step_t, vars)
    k3 = f(v + 0.5 * k2 / step_t, vars)
    k4 = f(v + k3 / step_t, vars)
    v1 = v + (k1 + 2 * k2 + 2 * k3 + k4) / (step_t * 6.0)

    # Check for negative values in all RK4 steps
    neg_mask = (
        (v1 < 0)
        | ((v + 0.5 * k1 / step_t) < 0)
        | ((v + 0.5 * k2 / step_t) < 0)
        | ((v + k3 / step_t) < 0)
    )

    if np.any(neg_mask):
        # v2 = np.log(np.clip(v, 1e-26, None))  # avoid log(0)
        v2 = np.log(v)  # avoid log(0)
        FT2 = flog(v2, vars)
        v2 = v2 + FT2 / step_t
        v1[neg_mask] = np.exp(v2[neg_mask])

    return v1


def call_function(v, Temp, Tmean, LAT, CC, egg_activate, step_t):
    diapause_lay = mosq_dia_lay(Tmean, LAT)
    diapause_hatch = mosq_dia_hatch(Tmean, LAT)
    ed_survival = mosq_surv_ed(Temp, step_t)

    shape_output = (
        v.shape[0],
        v.shape[1],
        5,
        int(Temp.shape[2] / step_t),
    )
    v_out = np.zeros(shape=shape_output)
    logger.debug(f"Shape v_out:{v_out.shape}")

    for t in range(Temp.shape[2]):
        #if t == 3:
        #    break

        T = Temp[:, :, t]
        birth = mosq_birth(T)
        dev_j = mosq_dev_j(T)
        dev_i = mosq_dev_i(T)
        dev_e = 1.0 / 7.1  # original model

        # Octave: ceil(t/step_t), Python: int(np.ceil((t+1)/step_t)) - 1
        idx_time = int(np.ceil((t + 1) / step_t)) - 1

        dia_lay = diapause_lay.values[:, :, idx_time]
        dia_hatch = diapause_hatch.values[:, :, idx_time]
        ed_surv = ed_survival[:, :, t]
        water_hatch = egg_activate.values[:, :, idx_time]
        mort_e = mosq_mort_e(T)
        mort_j = mosq_mort_j(T)
        Tmean_slice = Tmean[:, :, idx_time]
        logger.info(f"Tmean shape: {Tmean.shape}")
        logger.info(f"Tmean_slice shape: {Tmean_slice.shape}")
        mort_a = mosq_mort_a(Tmean_slice)

        # Add this block:
        logger.debug(f"Time step {t}:")
        logger.debug(f"  T.shape: {T.shape}")
        logger.debug(f"  birth.shape: {getattr(birth, 'shape', None)}")
        logger.debug(f"  dia_lay.shape: {getattr(dia_lay, 'shape', None)}")
        logger.debug(f"  mort_e.shape: {getattr(mort_e, 'shape', None)}")
        logger.debug(f"  water_hatch.shape: {getattr(water_hatch, 'shape', None)}")
        logger.debug(f"  dev_e: {dev_e}")
        logger.debug(f"  v[..., 0].shape: {v[..., 0].shape}")

        vars_tuple = (
            idx_time + 1,  # Octave uses 1-based, so pass idx_time+1
            step_t,
            Temp,
            CC.values,
            birth.values,
            dia_lay,
            dia_hatch,
            mort_e.values,
            mort_j.values,
            mort_a.values,
            ed_surv.values,
            dev_j.values,
            dev_i.values,
            dev_e,
            water_hatch,
        )

        va = rk4_step(eqsys, eqsys_log, v, vars_tuple, step_t)
        logger.info(f"Time step: {t}")


        # # Zero compartment 2 (Python index 1) if needed
        if (t / step_t) % 365 == 200:
            va[..., 1] = 0

        # Store output every step_t
        logger.info(f"time step: {t+1}")
        if (t + 1) % step_t == 0:            
            logger.info(f"IDX TIME: {idx_time}")
            if ((idx_time) % 30) == 0:
                logger.debug(f"MOY: {int(((t)/step_t) / 30)}")
            for j in range(5):
                v_out[..., j, idx_time] = np.maximum(va[..., j], 0)

    return v_out
