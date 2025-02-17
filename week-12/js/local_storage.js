"use strict";
const ls = {
    PREFIX: "judge0.",
    set(key, value) {
        if (!key) {
            return;
        }

        try {
            if (value == null) {
                ls.del(key);
                return;
            }

            if (typeof value === "object") {
                value = JSON.stringify(value);
            }

            localStorage.setItem(`${ls.PREFIX}${key}`, value);
        } catch (ignorable) {
        }
    },
    get(key) {
        if (!key) {
            return null;
        }

        try {
            const value = localStorage.getItem(`${ls.PREFIX}${key}`);
            try {
                return JSON.parse(value);
            } catch (ignorable) {
                return value;
            }
        } catch (ignorable) {
            return null;
        }
    },
    del(key) {
        if (!key) {
            return;
        }

        try {
            localStorage.removeItem(`${ls.PREFIX}${key}`);
        } catch (ignorable) {
        }
    }
};

export default ls;
