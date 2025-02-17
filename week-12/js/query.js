"use strict";
const query = {
    get(key) {
        const query = window.location.search.substring(1);
        const vars = query.split("&");
        for (let i = 0; i < vars.length; i++) {
            const pair = vars[i].split("=");
            if (decodeURIComponent(pair[0]) == key) {
                return decodeURIComponent(pair[1]);
            }
        }
    },
    keys() {
        const query = window.location.search.substring(1);
        const vars = query.split("&");
        const keys = [];
        for (let i = 0; i < vars.length; i++) {
            const pair = vars[i].split("=");
            keys.push(decodeURIComponent(pair[0]));
        }
        return keys;
    }
};

export default query;
