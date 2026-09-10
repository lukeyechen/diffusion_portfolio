import 'package:flutter/widgets.dart';

import 'google_sign_in_button_native.dart'
    if (dart.library.html) 'google_sign_in_button_web.dart' as platform;

Widget buildGoogleSignInButton({
  required Future<void> Function() onPressed,
}) {
  return platform.buildGoogleSignInButton(onPressed: onPressed);
}
