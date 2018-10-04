'use strict';

// Make sure we are in strict mode.
(function() {
  var strictMode = false;
  try {
    NaN = NaN;
  } catch (err) {
    strictMode = true;
  }
  if (!strictMode) {
    throw 'Unable to activate strict mode.';
  }
})();

// Make sure the content script is only run once on the page.
if (!window.hashpassLoaded) {
  window.hashpassLoaded = true;
  
  // Stores a document inside of which activeElement is located.
  var activeDocument = document;

  // Register the message handler.
  chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
      // Trims the attribute and converts it to lowercase.
      var normalizeAttr = function(attr) {
        return attr.replace(/^\s+|\s+$/g, '').toLowerCase();
      };

      // Checks if activeElement is inside iframe or not and returns correct document.
      var getActiveDocument = function() {
        var elem = document.activeElement;
        if (normalizeAttr(elem.tagName) === normalizeAttr('iframe')) {
          return elem.contentDocument;
        }
        return document;
      };

      // Returns whether elem is an input of type "password".
      var isPasswordInput = function(elem) {
        if (elem) {
          if (normalizeAttr(elem.tagName) === normalizeAttr('input')) {
            if (normalizeAttr(elem.type) === normalizeAttr('password')) {
              return true;
            }
          }
        }
        return false;
      };

      // Check if a password field is selected.
      if (request.type === 'hashpassCheckIfPasswordField') {
        activeDocument = getActiveDocument();
        if (activeDocument && isPasswordInput(activeDocument.activeElement)) {
          activeDocument.activeElement.style.boxShadow="#126c98 0px 0px 22px";
          
          sendResponse({ type: 'password' });
          return;
        }
        else
        {
          var pass_fields = ['pass', 'password'];
          
          for (var i in pass_fields) {
            if (activeDocument.getElementsByName(pass_fields[i]).length == 1) {
              var password_field = activeDocument.getElementsByName(pass_fields[i])[0];
              
              if (password_field.style.display!="none" && password_field.style.visibility!="hidden") {
                  password_field.style.boxShadow="#126c98 0px 0px 22px";
                  password_field.focus();
                  
                  sendResponse({ type: 'password' });
                  return;
              }
            }
          }
        }
        sendResponse({ type: 'not-password' });
        return;
      }
      
      // Get username
      if (request.type === 'getUserName') {
        activeDocument = getActiveDocument();
        if (activeDocument) {
          var user_fields = ['user', 'identifier', 'username'];

          for (var i in user_fields) {
            if (activeDocument.getElementsByName(user_fields[i]).length > 0) {
              var username_field = activeDocument.getElementsByName(user_fields[i])[0];
              
              sendResponse({ username: username_field.value });
              return;
            }
          }
        }
        sendResponse({ username: '' });
        return;
      }

      // Fill in the selected password field.
      if (request.type === 'hashpassFillPasswordField') {
        if (isPasswordInput(activeDocument.activeElement)) {
          activeDocument.activeElement.value = request.hash;
          
          // simulate a keyup event for websites inteested in it
          var event = new Event('keyup');
          activeDocument.activeElement.dispatchEvent(event);
          
          // simulate a change event for websites inteested in it
          var event = new Event('change');
          activeDocument.activeElement.dispatchEvent(event);
          
          activeDocument.activeElement.style.boxShadow="#126c98 0px 0px 22px";

          sendResponse({ type: 'close' });
          return;
        }
        sendResponse({ type: 'fail' });
        return;
      }
    }
  );
}
