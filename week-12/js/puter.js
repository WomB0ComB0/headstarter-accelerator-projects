"use strict";

export const IS_PUTER = puter.env === "app";

export function usePuter() {
    return IS_PUTER || puter.auth.isSignedIn();
}

async function uiSignIn() {
    document.getElementById("judge0-sign-in-btn").classList.add("judge0-hidden");
    const signOutBtn = document.getElementById("judge0-sign-out-btn");
    signOutBtn.classList.remove("judge0-hidden");
    signOutBtn.querySelector("#judge0-puter-username").innerText = (await puter.auth.getUser()).username;

    const modelSelect = document.getElementById("judge0-chat-model-select");
    modelSelect.closest(".ui.selection.dropdown").classList.remove("disabled");

    const userInput = document.getElementById("judge0-chat-user-input");
    userInput.disabled = false;
    userInput.placeholder = `Message ${modelSelect.value}`;

    document.getElementById("judge0-chat-send-button").disabled = false;
    document.getElementById("judge0-inline-suggestions").disabled = false;
}

function uiSignOut() {
    document.getElementById("judge0-sign-in-btn").classList.remove("judge0-hidden");
    const signOutBtn = document.getElementById("judge0-sign-out-btn");
    signOutBtn.classList.add("judge0-hidden");
    signOutBtn.querySelector("#judge0-puter-username").innerText = "Sign out";

    const modelSelect = document.getElementById("judge0-chat-model-select");
    modelSelect.closest(".ui.selection.dropdown").classList.add("disabled");

    const userInput = document.getElementById("judge0-chat-user-input");
    userInput.disabled = true;
    userInput.placeholder = `Sign in to chat with ${modelSelect.value}`;

    document.getElementById("judge0-chat-send-button").disabled = true;
    document.getElementById("judge0-inline-suggestions").disabled = true;
}

function updateSignInUI() {
    if (puter.auth.isSignedIn()) {
        uiSignIn();
    } else {
        uiSignOut();
    }
}

async function signIn() {
    await puter.auth.signIn();
    updateSignInUI();
}

function signOut() {
    puter.auth.signOut();
    updateSignInUI();
}

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("judge0-sign-in-btn").addEventListener("click", signIn);
    document.getElementById("judge0-sign-out-btn").addEventListener("click", signOut);
    updateSignInUI();
});
