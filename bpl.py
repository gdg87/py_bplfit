#Power law binned module
import math
import numpy
import scipy

def bplpva(h, boundaries, **kwargs):
	rngal = []
	limit = []
	bminb = []

	# ---------------------------------------------------------------
	# ---------------Parsing command-line arguments------------------
	# ---------------------------------------------------------------
	i=1
	for key, value in kwargs.iteritems():
		if key == 'range':
			rngal = value
		elif key == 'limit':
			limit = value
		elif key == 'bmin':
			bminb = value
		else:
			print "Ignoring invalid argument %s" % (key)

	# ---------------------------------------------------------------
	# ------------------------Checking input-------------------------
	# ---------------------------------------------------------------

	# 1. h must have integer counts.
	if all(type(x) is int for x in h) == False:
		print "(BPLFIT) Error: Vector h should be an integer vector"
    	alpha = float('nan')
    	bmin = boundaries[0]
    	L = float('nan')
    	return

	# 2. h must be non-negative
	if len(filter(lambda x: x < 0, h)):
    	print "(BPLFIT) Error: Vector h should be non-negative"
    	alpha = float('nan')
    	bmin = boundaries[0]
    	L = float('nan')
    	return

	# 3. boundaries must have number of elements as one more than the number in h
	if len(boundaries) != len(h) + 1:
		print "(BPLFIT) Error: Incorrect number of elements in either boundaries or h"
		alpha = float('nan')
    	bmin = boundaries[0]
    	L = float('nan')
		return

	# 4. Need atleast 2 bins to work with.
	if len(h) < 2:
		print "(BPLFIT) Error: I need atleast 2 bins to make this work"
		alpha = float('nan')
    	bmin = boundaries[0]
    	L = float('nan')
		return

	# 5. Checking range vector
	if rngal and (not isinstance(rngal, list) or min(rngal or [0]) < 1):
		print "(BPLFIT) Error: 'range' argument must contain a valid vector; using default"
		rngal = numpy.arange(1.5, 3.51, 0.01) #rngal = 1.5:0.01:3.5;
		return

	# 6. Checking limit option
	if limit and (not isinstance(limit, int) or limit < min(boundaries)):
		print "(BPLPVA) Error: 'limit' argument must be a positive value >= boundaries(1); using default."
		limit = boundaries[-2] #limit = boundaries(end-2);
		return

	# 7. Checking bmin option
	if bminb and (not isinstance(bminb, int) or bminb >= boundaries[-1]):
		print "(BPLPVA) Error: 'bmin' argument must be a positive value < boundaries(end-1); using default."
		bminb = boundaries[0]
		return

#PValue
def bplpva(h, boundaries, bmin, **kwargs):
	rngal = []
	limit = []
	bminb = []
	reps = 1000
	silent = 0

	# Parsing arguments
	for key, value in kwargs.iteritems():
		if key == 'range':
			rngal = value
		elif key == 'bminb':
			bminb = value
		elif key == 'limit':
			limit = value
		elif key == 'reps':
			reps = value
		elif key == 'silent':
			silent = 1
		else:
			print "Ignoring invalid argument %s" % (key)

	#Check input
	# 1. h must have integer counts.
	if all(type(x) is int for x in h) == False:
		print "(BPLPVA) Error: Vector h should be an integer vector"
		return

	# 2. h must be non-negative
	if len(filter(lambda x: x < 0, h)):
		print "(BPLPVA) Error: Vector h should be non-negative"
		return
	# 3. boundaries must have number of elements as one more than the number in h
	if len(boundaries) != len(h) + 1:
		print "(BPLPVA) Error: Incorrect number of elements in either boundaries or h"
		return
	# 4. Need atleast 2 bins to work with.
	if len(h) < 2:
		print "(BPLPVA) Error: I need atleast 2 bins to make this work"
		return
	# 5. Checking range vector
	if rngal and (not isinstance(rngal, list) or min(rngal or [0]) < 1):
		print "(BPLPVA) Error: 'range' argument must contain a valid vector; using default"
		rngal = numpy.arange(1.5, 3.51, 0.01) #rngal = 1.5:0.01:3.5;
		return
	# 6. Checking limit option
	if limit and (not isinstance(limit, int) or limit < min(boundaries)):
		print "(BPLPVA) Error: 'limit' argument must be a positive value >= boundaries(1); using default."
		limit = boundaries[-2] #limit = boundaries(end-2);
		return
	# 7. Checking bmin option
	if bminb and (not isinstance(bminb, int) or bminb >= boundaries[-1]):
		print "(BPLPVA) Error: 'bmin' argument must be a positive value < boundaries(end-1); using default."
		bminb = boundaries[0]
		return
	# 8. Checking number of repititons
	if reps and (not isinstance(reps, int) or reps < 2):
		print "(BPLPVA) Error: ''reps'' argument must be a positive value > 1; using default"
		reps = 1000
		return

	# Reshape the input vectors
	h = numpy.reshape(h, (len(h), 1))
	boundaries = numpy.reshape(boundaries, (len(boundaries), 1))

	N = sum(h)
	d = numpy.zeros((reps,1))

	if  not silent:
		print "Power-law distribution, parameter uncertainty calculation"
    	print "Copyright 2012 Yogesh Virkar";
    	print "Warning: This can be a slow calculation; please be patient"
    	print "reps = %i" % len(d);

    # ---------------------------------------------------------------
	#---------------Compute the empirical distance D*------------------
	# ---------------------------------------------------------------
	# Data above bmin
	ind = (boundaries>=bmin).argmax()
	z = h[ind:]
	nz = sum(z);
	b = boundaries[ind:];
	l = b[1:-1];
	u = b[2:];

	# Data below bmin
	y = h[1:ind-1];     #ny = sum(y);
	by = boundaries[1:ind];
	ly = by[1:-1];
	uy = by[2:];

	# Compute alpha using numerical maximization
	hnd = lambda alpha: (alpha) -sum( z*( numpy.log((l)**(1-alpha) - (u)**(1-alpha)) + (alpha-1)*numpy.log(bmin) ) );
	alpha = scipy.optimize.fmin(func=hnd, x0=1) #alpha = fminsearch(hnd, 1);

	# Compute distance using KS statistic
	temp = z.reverse().cumsum(axis=0)    #cumsum(z(end:-1:1));
	cx = 1 - temp.reverse()/nz    #1 - temp(end:-1:1)./nz;
	cf = 1 - numpy.power((l/bmin),(1-alpha))    #1 - (l./bmin).^(1-alpha);
	Dstar = concatenate(abs(cf - cx)).max() 	#max(abs(cf-cx));

	# ---------------------------------------------------------------
	# Compute the distribution of gofs using semiparametric bootstrap
	# ---------------------------------------------------------------
	# Probability of choosing value above bmin
	pz = nz/N;
	for i in range(1,reps + 1):
		#semi-parametric bootstrap of data
		n1 = sum(numpy.random.random((N,))>pz)
		temp = (ly+uy)/2
		temp2=[]
		for t in range(1, len(y) + 1):
			temp2 = numpy.array([temp2,kron(ones((y[t],1)),temp[t])])	#[temp2;repmat(temp(t),y(t),1)];
		temp2 = temp2[np.random.permutation(temp2.size())]	#temp2(randperm(numel(temp2)));
		x1 = temp2[ numpy.ceil(temp2.size() * numpy.random.random((1,n1)))]		#x1 = temp2(ceil(numel(temp2)*rand(n1,1)));
		n2 = N - n1;
		x2 = bmin*numpy.power(1 - numpy.random.random((1,n2), (-1/(alpha-1))))	#x2 = bmin.*(1-rand(n2,1)).^(-1/(alpha-1));
		x = [x1,x2]
		h2 = np.digitize(x, boundaries) 	#h2 = histc(x, boundaries);
		h2 = np.delete(h2, -1)	#h2(end) = [];
		ind = (h2.reverse() != 0).argmax() - 1    #ind = find(h2(end:-1:1)~=0,1,'first')-1;
		if ind == 1:
			h2 = np.delete(h2, -1)	#h2(end)= [];
		else:
			if ind > 1:
				ind2 = ind - 1
				end = h2.size() - 1
			h2 = np.delete(h2, range(end - ind2, end)) 	#h2(end-ind2:end) = [];
		boundaries2 = boundaries[1:-1-ind]; 	#boundaries2 = boundaries(1:end-ind);

    	# Need a minimum of 2 bins.
    	bmins = boundaries2[1:-1-2];
    	if bminb:
        	bmins = bmins[(bmins <= bminb).reverse().argmax()]	#bmins(find(bmins<=bminb, 1, 'last'));
    	if limit:
        	bmins[(bmins > limit)] = [] 	#bmins(bmins>limit) = [];
    	dat = zeros((a.shape[0], a.shape[1])) 	#zeros(size(bmins));

    	for xm in range(1,bmins.size() + 1):
    		bminq = bmins[xm];

    		# Truncate the data below bmin
        	indq = 	(boundaries2 >= bminq)	#find(boundaries2>=bminq, 1);
        	zq = h2[indq:-1];
        	nq = zq.sum(axis=0)	#sum(zq);
        	bq = boundaries2[indq:-1]	#boundaries2(indq:end);

        	# estimate alpha using specified range or using
        	# numerical maximization
        	lq = bq[1:-2]	#bq(1:end-1);
        	uq = bq[2:-1]	#bq(2:end);
        	if rngal:
        		H = kron(ones((1, size(rngal))), zp)	#repmat(zq, 1, numel(rngal));
        		LOWER_EDGE = kron(ones((1, size(rngal))), lq)	#repmat(lq, 1, numel(rngal));
        		UPPER_EDGE = kron(ones((1, size(rngal))), uq)	#repmat(uq, 1, numel(rngal));
        		ALPHA_EST = kron(ones((size(bq) - 1, 1)), rngal)	#repmat(rngal, numel(bq)-1, 1);
        		tempq = H * ( math.log(numpy.power(LOWER_EDGE, 1 - ALPHA_EST) - numpy.power(UPPER_EDGE, 1 - ALPHA_EST)) + (ALPHA_EST - 1) * math.log(bminq))  	#H .* (log(LOWER_EDGE.^(1-ALPHA_EST) - UPPER_EDGE.^(1-ALPHA_EST)) + (ALPHA_EST-1) .* log(bminq));
        		sum_ = sum(tempq, axis = 1)	#sum(tempq, 1);
        		I = sum_.index(max(sum_))	#[~,I] = max(sum_);
        		al = rngal[I]
        	else:
        		hnd = lambda al2: (al2) -sum( zq*( numpy.log((lq)**(1-al2) - (uq)**(1-al2)) + (al2-1)*numpy.log(bminq) ) ); #TODO
        		al = scipy.optimize.fmin(func=hnd, x0=1)	#fminsearch(hnd, 1);

            # compute KS statistic
        	tempq = cumsum(zq.reverse())	#cumsum(zq(end:-1:1));
        	cxq = 1 - tempq.reverse()/nq	#1 - tempq(end:-1:1)./nq;
        	cfq = 1 - numpy.power(lq/bminq, 1 - al) #1 - (lq./bminq).^(1-al);
        	dat[xm] = max(abs(cfq - cxq))	#dat(xm) = max(abs(cfq-cxq));

        if not silent:
        	print "iter = {}".format(i)

    	d[i] = min(dat)	#d(i) = min(dat);

	p = sum(d >= Dstar)/reps	#sum(d>=Dstar)./reps;

	return [p,d]
