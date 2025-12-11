function v = FE(func, func2, v, vars, step_t)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Forward Euler
    
    FT = func(v, vars);
	v1 = v .+  FT / step_t;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Check for negative numbers

	if any(any(v1 < 0))
	    v2 = log(v);
	    FT2 = func2(v2, vars);
		v2(v1<0) = v2(v1<0) .+ FT2(v1<0) ./ step_t;
	    v1(v1<0) = exp(v2(v1<0));
    endif

    v = v1;

endfunction
