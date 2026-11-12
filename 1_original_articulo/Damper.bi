model Damper {
	/* Specifying all variables in the problem */
	const T_s = .1, m = 2           /* Known constants */
	param f_c, c_0, k, p            /* Unknown variables */
	noise v                         /* Process noise */
	state s, sdot, s_old, sdot_old  /* State variables */
	obs y

	sub parameter {
		/* Specifying all priors for the unknown parameters  */
		f_c ~ gamma(2, 0.01)
		c_0 ~ gamma(2, 1)
		k ~ gamma(4, 0.3)
		p ~ uniform(0, 1)
	}
	
	sub proposal_parameter {
		/* Specifying the random walk proposal used in PMH */
		f_c ~ truncated_gaussian(f_c, 1.0e-3, 0.0)
		c_0 ~ truncated_gaussian(c_0, 1.0e-2, 0.0, 1.0)
		k ~ truncated_gaussian(k, 1.0e-2, 0.0)
		p ~ truncated_gaussian(p, 1.0e-2, 0.0, 1.0)
	}

	sub initial {
		/* Specifying the inital state variables */
		s <- 0.5
		sdot <- 0.0
	}

	sub transition(delta = 1) {
		/* Describe how the states in the model evolves at each time step. The variables s_old and sdot_old is only for temprary storage of the previous state values. */
		v ~ gaussian(0.0, 1.0e-2)
		
		s_old <- s
		sdot_old <- sdot
		
		s <- s_old + T_s*(sdot_old)
		sdot <- sdot_old + (T_s/m)*(-f_c*((sdot_old < 0) ? -1 : 1) - c_0*sdot_old - k*((s_old < 0) ? -1 : 1)*pow(abs(s_old), p)) + v
	}

	sub observation {
		/* Describe how the measurements relates to the states */
		y ~ gaussian(s, 1.0e-1)
	}
}