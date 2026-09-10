import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';

class GoogleAuthController extends ChangeNotifier {
  static const serverClientId = String.fromEnvironment(
    'GOOGLE_SERVER_CLIENT_ID',
    defaultValue:
        '144819245123-2p7ck8l2jonk73sulhn67hhfbj5ti9ua.apps.googleusercontent.com',
  );

  final GoogleSignIn _signIn = GoogleSignIn.instance;
  StreamSubscription<GoogleSignInAuthenticationEvent>? _subscription;
  GoogleSignInAccount? _user;
  String? _idToken;
  String? _error;
  bool _busy = false;
  bool _initialized = false;

  bool get isConfigured => serverClientId.trim().isNotEmpty;
  bool get isBusy => _busy;
  bool get isSignedIn => _user != null && _idToken != null;
  GoogleSignInAccount? get user => _user;
  String? get idToken => _idToken;
  String? get error => _error;

  Future<void> initialize() async {
    if (_initialized || !isConfigured) {
      return;
    }
    _initialized = true;
    _setBusy(true);
    try {
      await _signIn.initialize(serverClientId: serverClientId);
      _subscription = _signIn.authenticationEvents.listen(
        _handleAuthenticationEvent,
        onError: _handleAuthenticationError,
      );
      final lightweight = _signIn.attemptLightweightAuthentication();
      if (lightweight != null) {
        final user = await lightweight;
        if (user != null) {
          _setUser(user);
        }
      }
    } on GoogleSignInException catch (error) {
      _error = _messageForException(error);
    } on Exception catch (error) {
      _error = 'Google sign-in could not start: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> signIn() async {
    if (!isConfigured) {
      _error = 'Google sign-in needs an OAuth client ID.';
      notifyListeners();
      return;
    }
    if (!_initialized) {
      await initialize();
    }
    _setBusy(true);
    _error = null;
    try {
      if (!_signIn.supportsAuthenticate()) {
        throw UnsupportedError('Interactive Google sign-in is unavailable.');
      }
      final user = await _signIn.authenticate();
      _setUser(user);
    } on GoogleSignInException catch (error) {
      _error = _messageForException(error);
    } on Exception catch (error) {
      _error = 'Google sign-in failed: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> signOut() async {
    _setBusy(true);
    try {
      await _signIn.signOut();
      _clearUser();
    } finally {
      _setBusy(false);
    }
  }

  void _handleAuthenticationEvent(GoogleSignInAuthenticationEvent event) {
    if (event is GoogleSignInAuthenticationEventSignIn) {
      _setUser(event.user);
    } else if (event is GoogleSignInAuthenticationEventSignOut) {
      _clearUser();
    }
  }

  void _handleAuthenticationError(Object error) {
    _clearUser(notify: false);
    _error = error is GoogleSignInException
        ? _messageForException(error)
        : 'Google sign-in failed: $error';
    notifyListeners();
  }

  void _setUser(GoogleSignInAccount user) {
    final token = user.authentication.idToken;
    if (token == null || token.isEmpty) {
      _clearUser(notify: false);
      _error = 'Google did not return an identity token. Check the OAuth setup.';
    } else {
      _user = user;
      _idToken = token;
      _error = null;
    }
    notifyListeners();
  }

  void _clearUser({bool notify = true}) {
    _user = null;
    _idToken = null;
    if (notify) {
      notifyListeners();
    }
  }

  void _setBusy(bool value) {
    _busy = value;
    notifyListeners();
  }

  String _messageForException(GoogleSignInException error) {
    return switch (error.code) {
      GoogleSignInExceptionCode.canceled => 'Google sign-in was canceled.',
      GoogleSignInExceptionCode.clientConfigurationError =>
        'Google sign-in configuration is incomplete.',
      GoogleSignInExceptionCode.interrupted =>
        'Google sign-in was interrupted. Please try again.',
      _ => 'Google sign-in failed: ${error.description ?? error.code.name}',
    };
  }

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    super.dispose();
  }
}
