/**
 * Content script for providing AI text completion functionality in text input fields
 * 
 * This content script adds real-time AI text completion suggestions to text inputs and textareas
 * across all web pages. It displays suggestions in a floating element that follows the input,
 * and allows users to accept suggestions using the Tab key.
 */

let currentTextarea: HTMLTextAreaElement | HTMLInputElement | null = null;
let currentHandler: ((event: Event) => void) | null = null;
let websiteContext: string | null = null;

export default defineContentScript({
  // Match pattern to run this content script on all URLs
  matches: ['<all_urls>'],
  
  /**
   * Main content script function that sets up the text completion functionality
   */
  main() {
    console.log('[Content Script] Initializing text completion functionality');
    console.log('[Content Script] Script loaded');
    console.log('[Content Script] Current URL:', window.location.href);

    // Initialize website context
    try {
      const metaTags = document.getElementsByTagName("meta");
      const titleTags = document.getElementsByTagName("title");

      // Ensure the metaTags and titleTags collections are non-empty
      if (metaTags.length === 0) {
        console.warn("No <meta> tags found on the page.");
      }

      if (titleTags.length === 0) {
        console.warn("No <title> tags found on the page.");
      }

      // Combine meta and title tags into a single string
      websiteContext =
        Array.from(metaTags)
          .map((tag) => tag.outerHTML)
          .join("") +
        Array.from(titleTags)
          .map((tag) => tag.outerHTML)
          .join("");

      console.log("Website context initialized:", websiteContext);
    } catch (error) {
      console.error("Error during context initialization:", error);
    }

    // Detect when a textarea gains focus
    document.addEventListener("focusin", (event) => {
      const target = event.target as HTMLElement;

      // Ensure target is a valid HTMLElement
      if (!(target instanceof HTMLElement)) {
        console.error("Target is not an HTMLElement:", target);
        return;
      }

      try {
        if (
          target &&
          (target.tagName === "TEXTAREA" ||
            target.tagName === "INPUT" ||
            target.isContentEditable)
        ) {
          currentTextarea = target as HTMLTextAreaElement | HTMLInputElement;
          currentHandler = setupTextareaListener(currentTextarea);

          // Skip input types for privacy if necessary
          if (
            target.tagName === "INPUT" &&
            (target.getAttribute('type') === "email" || target.getAttribute('type') === "password")
          ) {
            console.log("Skipping input type for privacy:", target.getAttribute('type'));
            return;
          }

          // Ensure textarea is not already wrapped
          if (!target.parentElement?.classList.contains("textarea-container")) {
            const container = document.createElement("div");
            container.classList.add("textarea-container");
            container.style.position = "relative";
            container.style.display = "inline-block";
            container.style.width = `${target.offsetWidth}px`; // Match textarea width
            container.style.height = `${target.offsetHeight}px`; // Match textarea height
            container.style.boxSizing = "border-box"; // Include padding/border

            // Insert container and move textarea inside
            target.parentElement?.insertBefore(container, target);
            container.appendChild(target);

            // Create the ghost suggestion text element
            const suggestionOverlay = document.createElement("div");
            suggestionOverlay.classList.add("suggestion-overlay");

            // Copy necessary styles from the textarea
            const computed = window.getComputedStyle(target);
            // Copy all styles from computed to suggestionOverlay dynamically
            for (let property of Array.from(computed)) {
              suggestionOverlay.style[property as any] =
                computed.getPropertyValue(property);
            }

            // Assign specific styles for suggestionOverlay
            Object.assign(suggestionOverlay.style, {
              position: "absolute",
              top: "0",
              left: "0",
              width: "100%",
              height: "100%",
              pointerEvents: "none", // Allow text input
              zIndex: "0", // Ensure overlay stays behind text
              overflowY: "auto", // Prevent overlay from overlapping
              lineHeight: computed.lineHeight,
              boxSizing: "border-box", // Ensure padding/border is accounted for
            });

            container.appendChild(suggestionOverlay);
            (target as any).suggestionOverlay = suggestionOverlay;

            // Match textarea background to prevent double text effect
            target.style.background = "transparent";
            target.style.position = "relative";
            target.style.zIndex = "1"; // Ensure input is above the overlay
            target.style.boxSizing = "border-box"; // Ensure padding/border is included

            // Clear overlay text whenever the user starts typing
            target.addEventListener("focus", () => {
              suggestionOverlay.textContent = ""; // Clear the suggestion when focus happens
            });

            // Ensure textarea is scrollable if text overflows
            target.style.overflowY = "auto";

            target.focus();
          }
        }
      } catch (error) {
        console.error("Error in focusin event listener:", error);
      }
    });

    // Detect when a textarea loses focus
    document.addEventListener("focusout", (event) => {
      if (event.target === currentTextarea) {
        // Check if currentTextarea and currentHandler are defined before attempting to remove the event listener
        if (currentTextarea) {
          if (currentHandler) {
            try {
              currentTextarea.removeEventListener("input", currentHandler);
              currentHandler = null; // Reset the handler after removal
              console.log("Removed input event listener successfully.");
            } catch (error) {
              console.error("Error removing input event listener:", error);
            }
          } else {
            console.warn("No currentHandler found to remove.");
          }

          currentTextarea = null; // Reset currentTextarea after cleanup
        } else {
          console.warn("No currentTextarea found on focusout.");
        }
      }
    });

    // Add suggestion when user presses Tab key
    document.addEventListener("keydown", (event) => {
      try {
        // Get the current text from the textarea. If it is contentEditable, use innerText instead of value.
        if (!currentTextarea) {
          return;
        }

        let currentText = currentTextarea.isContentEditable
          ? currentTextarea.innerText ?? ""
          : (currentTextarea as HTMLInputElement | HTMLTextAreaElement).value ?? "";

        let superKey = "⌘";
        if (!/(Mac|iPhone|iPod|iPad)/i.test(navigator.platform)) {
          superKey = "Ctrl";
        }

        // Allow superKey keybinds to go through
        if (event.key === superKey && (event.ctrlKey || event.metaKey)) {
          return;
        }

        if (
          (currentTextarea as any).suggestionOverlay &&
          (event.key === "Escape" ||
            event.key === "Enter" ||
            event.key === "Backspace")
        ) {
          (currentTextarea as any).suggestionOverlay.innerHTML = "";
          adjustTextHeights(currentTextarea);
        }

        if (
          event.key === "Tab" &&
          currentTextarea &&
          (currentTextarea as any).suggestionOverlay
        ) {
          event.preventDefault(); // Prevent default Tab behavior immediately.

          let ghostSpan =
            (currentTextarea as any).suggestionOverlay.querySelector(".ghost-text");

          // Check if ghostSpan exists
          if (!ghostSpan) {
            console.error("No ghost-text span found in suggestion overlay.");
            return;
          }

          let suggestionText = ghostSpan.textContent || "";

          let updatedText = currentText + suggestionText;

          if (currentTextarea.isContentEditable) {
            currentTextarea.textContent = updatedText;
          } else {
            (currentTextarea as HTMLInputElement | HTMLTextAreaElement).value = updatedText.replace(/\n$/, ""); // Prevent unintended newlines.
          }

          // Update the overlay and adjust layout.
          (currentTextarea as any).suggestionOverlay.innerHTML = escapeHTML(updatedText);
          moveCursorToEnd(currentTextarea);
          adjustTextHeights(currentTextarea);
        }
      } catch (error) {
        console.error("Error during keydown event handling:", error);
      }
    });
  }
});

// Listen for input events with debouncing
function setupTextareaListener(textarea: HTMLElement) {
  let debounceTimeout: number;

  if (!textarea) {
    console.error("No textarea provided to setupTextareaListener");
    return () => {};
  }

  const handleInput = () => {
    // Clear any pending timeout
    clearTimeout(debounceTimeout);

    if (!currentTextarea) {
      console.error("No currentTextarea found for input event");
      return;
    }

    // Clear the suggestion overlay when the user starts typing
    if ((currentTextarea as any).suggestionOverlay) {
      (currentTextarea as any).suggestionOverlay.innerHTML = "";
    } else {
      console.error("suggestionOverlay is not available");
      return;
    }

    // Set new timeout to fire after 1 second of inactivity
    debounceTimeout = window.setTimeout(() => {
      // Only send if the textarea is still focused
      if (document.activeElement === textarea) {
        // contentEditable divs don't have value, so need to use innerText
        let currentText =
          currentTextarea && currentTextarea.isContentEditable
            ? currentTextarea.innerText
            : (currentTextarea as HTMLInputElement | HTMLTextAreaElement)?.value ?? "";

        if (!currentText || currentText.trim() === "") {
          return;
        }

        try {
          browser.runtime.sendMessage({
            type: "TEXTAREA_UPDATE",
            value: currentText,
            context: websiteContext,
          }).then(response => {
            if (response?.success && (currentTextarea as any)?.suggestionOverlay) {
              console.log("received suggestion", response.result);
              const suggestion = response.result;

              // Build overlay content:
              // The user text is rendered normally and the suggestion is in a span with a lighter color.
              try {
                (currentTextarea as any).suggestionOverlay.innerHTML =
                  '<span class="user-text">' +
                  escapeHTML(currentText) +
                  "</span>" +
                  '<span class="ghost-text" style="font-style: italic;">' +
                  escapeHTML(suggestion) +
                  "</span>";

                adjustTextHeights(currentTextarea as HTMLElement, suggestion); // Adjust height to fit suggestion
              } catch (innerError) {
                console.error(
                  "Error updating suggestion overlay:",
                  innerError
                );
              }
            } else {
              console.error("Failed to receive valid response:", response);
            }
          }).catch(error => {
            console.error("Error in message response:", error);
          });
        } catch (error) {
          console.error("Error sending message:", error);
        }
      }
    }, 1000);
  };

  textarea.addEventListener("input", handleInput);
  return handleInput; // Return the handler for cleanup
}

function moveCursorToEnd(textarea: HTMLElement) {
  try {
    // Focus the textarea
    textarea.focus();

    // Handle contentEditable
    if (textarea.isContentEditable) {
      const range = document.createRange();
      const selection = window.getSelection();
      if (selection) {
        range.selectNodeContents(textarea);
        range.collapse(false);

        // Clear any previous selections and apply the new range
        selection.removeAllRanges();
        selection.addRange(range);
      }
    } else {
      // For non-contentEditable, set the selection range to the end
      const input = textarea as HTMLInputElement | HTMLTextAreaElement;
      input.setSelectionRange(input.value.length, input.value.length);
    }
  } catch (error) {
    console.error("Error moving cursor to end:", error);
  }
}

// Adjust the height of the textarea to fit the content
function adjustTextHeights(textarea: HTMLElement, suggestion = "") {
  try {
    // Save the current value of the textarea to avoid altering user input
    const originalValue = (textarea as HTMLInputElement | HTMLTextAreaElement).value;

    // If there is a suggestion, temporarily add it to calculate the height
    if (suggestion !== "") {
      // Create a temporary element to hold the suggestion text for height calculation
      const tempDiv = document.createElement("div");
      tempDiv.style.position = "absolute"; // Ensure it's out of the flow
      tempDiv.style.visibility = "hidden"; // Make it invisible
      tempDiv.style.whiteSpace = "pre-wrap"; // Preserve line breaks, if any
      tempDiv.style.wordWrap = "break-word"; // Prevent word overflow
      tempDiv.style.font = window.getComputedStyle(textarea).font; // Match textarea font

      // Append the suggestion text to the tempDiv
      tempDiv.textContent = suggestion;

      document.body.appendChild(tempDiv);

      // Set the textarea height to auto before measuring
      textarea.style.height = "auto";

      // Adjust the height based on the content and the suggestion
      const newHeight = Math.max(textarea.scrollHeight, tempDiv.scrollHeight);
      textarea.style.height = newHeight + "px";

      // Clean up after measuring
      document.body.removeChild(tempDiv);
    } else {
      // Reset the height to fit the content only (no suggestion)
      textarea.style.height = "auto"; // Ensure textarea resizes to content
      textarea.style.height = textarea.scrollHeight + "px";
    }

    // Adjust the parent element height to match textarea
    if (textarea.parentElement) {
      textarea.parentElement.style.height = textarea.style.height;
    }

    moveCursorToEnd(textarea); // Ensure the cursor stays at the end after resizing
  } catch (error) {
    console.error("Error adjusting text heights:", error);
  }
}

function escapeHTML(str: string) {
  try {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  } catch (error) {
    console.error("Error escaping HTML:", error);
    return str; // If error occurs, return the original string to prevent breaking the code
  }
}
