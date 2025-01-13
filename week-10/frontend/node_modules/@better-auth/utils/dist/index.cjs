'use strict';

const uncrypto = require('uncrypto');



Object.keys(uncrypto).forEach(function (k) {
	if (k !== 'default' && !Object.prototype.hasOwnProperty.call(exports, k)) exports[k] = uncrypto[k];
});
